"""
Word Association Network Analysis with SRF
==========================================

Paper-faithful implementation of SWOW graph construction with NaN preservation
for demonstrating SRF's unique capability to handle explicit missing data.

Based on createSWOWGraph.R from SWOWEN-2018 repository.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import NMF
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

from pysrf import SRF


def filter_by_node_degree_postgraph(G_directed, top_n_words=3000):
    """
    Filter graph to top N words by degree AFTER graph construction.
    This ensures you get exactly N words in the final network.
    """
    # Calculate degree in the directed graph
    # node_degrees = dict(G_directed.degree())  # in-degree + out-degree
    # Calculates WEIGHTED degree (sum of edge weights)
    node_degrees = dict(G_directed.degree(weight="weight"))

    # Get top N nodes
    top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:top_n_words]

    print(f"\nTop 10 nodes by degree in graph:")
    for node in top_nodes[:10]:
        print(f"  {node}: {node_degrees[node]} connections")

    # Create subgraph with only top nodes
    G_filtered = G_directed.subgraph(top_nodes).copy()

    print(
        f"\nFiltered graph: {G_filtered.number_of_nodes()} nodes, "
        f"{G_filtered.number_of_edges()} edges"
    )

    return G_filtered


def load_swow_data_paper_format(data_dir, use_all_responses=True):
    """
    Load SWOW strength data (preprocessed association strengths).

    Parameters
    ----------
    data_dir : Path
        Directory containing SWOW strength files
    use_all_responses : bool
        If True, load R123 (all 3 responses). If False, load R1 only.
    top_n_cues : int, optional
        If provided, keep only top N cues by total outgoing strength

    Returns
    -------
    df : DataFrame
        Long format data with columns [cue, response, count, N, strength]
    """
    data_dir = Path(data_dir)

    if use_all_responses:
        strength_file = data_dir / "strength.SWOW-EN.R123.20180827.csv"
    else:
        strength_file = data_dir / "strength.SWOW-EN.R1.20180827.csv"

    df = pd.read_csv(strength_file, sep="\t")

    # Standardize column names
    if "R123.Strength" in df.columns:
        df["strength"] = df["R123.Strength"]
        df["count"] = df["R123"]
    elif "R1.Strength" in df.columns:
        df["strength"] = df["R1.Strength"]
        df["count"] = df["R1"]
    else:
        raise ValueError("Cannot find strength column in SWOW data")

    return df


def build_directed_graph(df_long, weight_col="strength"):
    """
    Build directed association graph following SWOW paper methodology.

    Implements the R function createGraph() and extractComponent():
    1. Create directed graph from cue-response pairs with weights
    2. Remove nodes with out-degree = 0 (responses that never act as cues)
    3. Remove self-loops
    4. Extract largest strongly connected component

    Parameters
    ----------
    df_long : DataFrame
        Long format data with columns [cue, response, strength]
    weight_col : str
        The name of the column to use as edge weight
        (e.g., 'strength' or 'ppmi_strength')

    Returns
    -------
    G_directed : nx.DiGraph
        Directed graph (largest strongly connected component)
    removed_nodes : set
        Set of nodes removed (not in largest SCC)
    """
    # Filter out missing responses
    df_filtered = df_long[df_long["response"].notna()].copy()

    # Create directed graph from edge list
    G = nx.DiGraph()

    # *** MODIFIED LINE ***
    # Use the specified weight column
    G.add_weighted_edges_from(
        df_filtered[["cue", "response", weight_col]].itertuples(index=False, name=None)
    )

    # Remove nodes with out-degree = 0 (responses that never appear as cues)
    nodes_to_remove = [node for node in G.nodes() if G.out_degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)

    # Remove self-loops (simplify)
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)

    # Extract largest strongly connected component
    strongly_connected = list(nx.strongly_connected_components(G))
    largest_scc = max(strongly_connected, key=len)

    G_directed = G.subgraph(largest_scc).copy()
    removed_nodes = set(G.nodes()) - largest_scc

    return G_directed, removed_nodes


def calculate_ppmi(df_long):
    """
    Calculates PPMI based on the paper's formula (Equation 2) and adds
    it to the DataFrame.

    PPMI(r|c) = max(0, log2( (p(r|c) * N) / sum_i(p(r|c_i)) ))

    Parameters
    ----------
    df_long : DataFrame
        Must have columns ['cue', 'response', 'strength']
        where 'strength' is p(r|c).

    Returns
    -------
    DataFrame
        Original DataFrame with a new 'ppmi_strength' column.
    """
    print("\nCalculating PPMI...")

    # N = number of unique cues
    N_cues = df_long["cue"].nunique()
    print(f"  N (unique cues): {N_cues}")

    # Calculate sum_i(p(r|c_i)) for each response 'r'
    # This is the marginal probability p(r) scaled by N
    print("  Calculating marginal response probabilities p(r)...")
    p_r_sum = df_long.groupby("response")["strength"].sum().reset_index()
    p_r_sum = p_r_sum.rename(columns={"strength": "p_r_sum"})

    # Merge this back into the main dataframe
    df_with_pr = pd.merge(df_long, p_r_sum, on="response", how="left")

    # Handle cases where a response might not have a p_r_sum (e.g., if it
    # never appears as a response, though this is unlikely in SWOW)
    df_with_pr["p_r_sum"] = df_with_pr["p_r_sum"].fillna(0)

    # Now calculate PPMI
    # p(r|c) = df_with_pr['strength']
    # N = N_cues
    # sum_i(p(r|c_i)) = df_with_pr['p_r_sum']

    # We need to handle division by zero or log(0)
    # If strength is 0, ratio is 0, log(0) = -inf, max(0, -inf) = 0
    # If p_r_sum is 0, ratio is inf, log(inf) = inf, max(0, inf) = inf
    # We use np.where to safely compute the log only for positive ratios

    numerator = df_with_pr["strength"] * N_cues
    denominator = df_with_pr["p_r_sum"]

    # Initialize ppmi column
    df_with_pr["ppmi_strength"] = 0.0

    # Calculate only where denominator is positive
    positive_denom_mask = denominator > 1e-10

    ratio = np.full(df_with_pr.shape[0], 0.0)
    ratio[positive_denom_mask] = (
        numerator[positive_denom_mask] / denominator[positive_denom_mask]
    )

    # Calculate log2 only where ratio is positive
    positive_ratio_mask = ratio > 1e-10
    log_ratio = np.full(df_with_pr.shape[0], -np.inf)
    log_ratio[positive_ratio_mask] = np.log2(ratio[positive_ratio_mask])

    # PPMI is max(0, log_ratio)
    df_with_pr["ppmi_strength"] = np.maximum(0, log_ratio)

    print(f"  PPMI calculation complete.")
    print("  Original strength vs. new PPMI strength (head):")
    print(df_with_pr[["cue", "response", "strength", "ppmi_strength"]].head())

    return df_with_pr


def symmetrize_with_nan_preservation(
    G_directed: nx.DiGraph,
) -> tuple[np.ndarray, list[str], dict]:
    vocabulary = sorted(G_directed.nodes())
    n = len(vocabulary)
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    A_directed = np.full((n, n), np.nan, dtype=np.float32)

    for source, target, data in G_directed.edges(data=True):
        i = word_to_idx[source]
        j = word_to_idx[target]
        A_directed[i, j] = data["weight"]

    A_symmetric = np.full((n, n), np.nan, dtype=np.float32)

    both_observed = 0
    one_observed = 0
    neither_observed = 0

    for i in range(n):
        print(f"Processing row {i} of {n}", end="\r")
        for j in range(i, n):
            if i == j:
                A_symmetric[i, j] = 0
                continue

            has_ij = ~np.isnan(A_directed[i, j])
            has_ji = ~np.isnan(A_directed[j, i])

            if has_ij and has_ji:
                A_symmetric[i, j] = (A_directed[i, j] + A_directed[j, i]) / 2
                A_symmetric[j, i] = A_symmetric[i, j]
                both_observed += 1
            elif has_ij:
                A_symmetric[i, j] = A_directed[i, j]
                A_symmetric[j, i] = A_directed[i, j]
                one_observed += 1
            elif has_ji:
                A_symmetric[i, j] = A_directed[j, i]
                A_symmetric[j, i] = A_directed[j, i]
                one_observed += 1
            else:
                neither_observed += 1

    n_nan = np.sum(np.isnan(A_symmetric))

    asymmetry_values = []
    for i in range(n):
        print(f"Processing asymmetry for row {i} of {n}", end="\r")
        for j in range(i + 1, n):
            if ~np.isnan(A_directed[i, j]) and ~np.isnan(A_directed[j, i]):
                asymm = abs(A_directed[i, j] - A_directed[j, i]) / (
                    A_directed[i, j] + A_directed[j, i] + 1e-10
                )
                asymmetry_values.append(asymm)

    metadata = {
        "n_words": n,
        "both_observed": both_observed,
        "one_observed": one_observed,
        "neither_observed": neither_observed,
        "pct_missing": n_nan / n**2 * 100,
        "mean_asymmetry": np.mean(asymmetry_values) if asymmetry_values else 0,
    }

    return A_symmetric, vocabulary, metadata


def fit_srf(network, rank=20):
    """Apply SRF to association network with NaN handling."""
    srf = SRF(
        rank=rank,
        rho=3.0,
        max_outer=600,
        max_inner=50,
        random_state=42,
        init="random_sqrt",
        tol=1e-4,
        verbose=1,
    )

    srf.fit(network)

    return srf.w_


def compute_dimension_coherence(W, network, top_n=50):
    """
    Measure semantic coherence of dimensions.

    Coherence = mean pairwise association strength among top words.
    """
    n_dims = W.shape[1]
    coherence_scores = np.zeros(n_dims)

    network_clean = np.nan_to_num(network, nan=0)

    for dim in range(n_dims):
        loadings = W[:, dim]
        top_idx = np.argsort(loadings)[-top_n:]

        similarities = []
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                sim = network_clean[top_idx[i], top_idx[j]]
                if sim > 0:
                    similarities.append(sim)

        if similarities:
            coherence_scores[dim] = np.mean(similarities)
        else:
            coherence_scores[dim] = 0

    return coherence_scores


def compute_sparsity(W):
    """Compute sparsity metrics for dimensions."""
    n_dims = W.shape[1]
    metrics = {
        "dimension": [],
        "gini": [],
        "entropy": [],
        "pct_active": [],
    }

    for dim in range(n_dims):
        loadings = W[:, dim]

        # Gini coefficient
        sorted_loadings = np.sort(loadings)
        n = len(loadings)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_loadings)) / (n * np.sum(sorted_loadings)) - (
            n + 1
        ) / n

        # Percentage active (above 10% of max)
        threshold = 0.1 * loadings.max()
        pct_active = 100 * np.sum(loadings > threshold) / n

        # Normalized entropy
        probs = loadings / (loadings.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(n)

        metrics["dimension"].append(dim)
        metrics["gini"].append(gini)
        metrics["entropy"].append(entropy)
        metrics["pct_active"].append(pct_active)

    return pd.DataFrame(metrics)


def get_top_words_per_dimension(W, vocabulary, top_n=20):
    """Extract top words for each dimension."""
    n_dims = W.shape[1]
    top_words_data = []

    for dim in range(n_dims):
        loadings = W[:, dim]
        top_idx = np.argsort(loadings)[-top_n:][::-1]

        for rank, idx in enumerate(top_idx):
            top_words_data.append(
                {
                    "dimension": dim,
                    "rank": rank,
                    "word": vocabulary[idx],
                    "loading": loadings[idx],
                }
            )

    return pd.DataFrame(top_words_data)


def plot_dimension_wordclouds(W, vocabulary, output_dir, n_dims=6, top_n=100):
    """Create wordclouds for semantic dimensions."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return

    # Color schemes for different dimensions
    colormaps = ["Blues", "Oranges", "Greens", "Reds", "Purples", "YlOrBr"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for dim in range(min(n_dims, W.shape[1])):
        ax = axes[dim]

        loadings = W[:, dim]
        top_idx = np.argsort(loadings)[-top_n:]

        word_weights = {
            vocabulary[idx]: loadings[idx] for idx in top_idx if loadings[idx] > 0
        }

        if word_weights:
            # Get top 5 words for title
            top_5 = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            top_words_str = ", ".join([w for w, _ in top_5])

            wc = WordCloud(
                width=600,
                height=400,
                background_color="white",
                colormap=colormaps[dim % len(colormaps)],
                relative_scaling=0.4,
                min_font_size=10,
                max_font_size=80,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(word_weights)

            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"Dimension {dim}\n{top_words_str}",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax.axis("off")
        else:
            ax.axis("off")

    for dim in range(n_dims, len(axes)):
        axes[dim].axis("off")

    plt.suptitle(
        "Semantic dimensions discovered by SRF",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / "plots" / "dimension_wordclouds.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_semantic_network(network, W, vocabulary, output_dir, dims=[0, 1], top_n=25):
    """Plot network graphs for dimensions."""
    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(10 * n_dims, 10))
    if n_dims == 1:
        axes = [axes]

    network_clean = np.nan_to_num(network, nan=0)

    for ax, dim in zip(axes, dims):
        loadings = W[:, dim]
        top_idx = np.argsort(loadings)[-top_n:]

        top_3_idx = np.argsort(loadings)[-3:][::-1]
        top_3_words = [vocabulary[i] for i in top_3_idx]

        G = nx.Graph()

        for i, idx_i in enumerate(top_idx):
            for j, idx_j in enumerate(top_idx):
                if i < j and network_clean[idx_i, idx_j] > 0.02:
                    G.add_edge(
                        vocabulary[idx_i],
                        vocabulary[idx_j],
                        weight=network_clean[idx_i, idx_j],
                    )

        node_sizes = [loadings[idx] * 3000 for idx in top_idx if vocabulary[idx] in G]
        node_colors = [loadings[idx] for idx in top_idx if vocabulary[idx] in G]
        nodes = [vocabulary[idx] for idx in top_idx if vocabulary[idx] in G]

        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

        edges = G.edges()
        weights = [G[u][v]["weight"] * 100 for u, v in edges]

        nx.draw_networkx_edges(
            G, pos, width=weights, alpha=0.2, edge_color="gray", ax=ax
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_size=node_sizes,
            node_color=node_colors,
            cmap="YlOrRd",
            alpha=0.9,
            ax=ax,
            vmin=0,
            vmax=loadings.max(),
        )

        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

        ax.set_title(
            f"Dimension {dim}: {', '.join(top_3_words)}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.axis("off")

    plt.suptitle(
        "Word association networks by semantic dimension",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    dims_str = "_".join(map(str, dims))
    output_path = output_dir / "plots" / f"semantic_networks_dims_{dims_str}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_loading_heatmap(W, vocabulary, output_dir, top_n_per_dim=20):
    """Plot heatmap of word loadings across dimensions."""
    n_dims = W.shape[1]

    top_words_per_dim = set()
    for dim in range(n_dims):
        top_idx = np.argsort(W[:, dim])[-top_n_per_dim:]
        top_words_per_dim.update(top_idx)

    top_idx_list = sorted(list(top_words_per_dim))
    W_subset = W[top_idx_list, :]
    word_labels = [vocabulary[i] for i in top_idx_list]

    fig, ax = plt.subplots(figsize=(12, max(8, len(top_idx_list) * 0.15)))

    sns.heatmap(
        W_subset,
        cmap="YlOrRd",
        yticklabels=word_labels,
        xticklabels=[f"Dim {i}" for i in range(n_dims)],
        cbar_kws={"label": "Loading"},
        ax=ax,
    )

    ax.set_title("Word loadings across dimensions", fontsize=14)
    ax.set_xlabel("Dimensions", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "plots" / "loading_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sparsity_distribution(W, output_dir):
    """Plot distribution of loadings showing sparsity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Distribution of loadings
    ax = axes[0]
    all_loadings = W.flatten()
    ax.hist(all_loadings, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    ax.axvline(
        all_loadings.mean(), color="red", linestyle="--", linewidth=2, label="Mean"
    )
    ax.set_xlabel("Loading value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of word loadings", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Sparsity per dimension
    ax = axes[1]
    n_dims = W.shape[1]
    pct_active = []
    for dim in range(n_dims):
        loadings = W[:, dim]
        threshold = 0.1 * loadings.max()
        pct = 100 * np.sum(loadings > threshold) / len(loadings)
        pct_active.append(pct)

    ax.bar(range(n_dims), pct_active, color="#2ecc71", alpha=0.7, edgecolor="black")
    ax.axhline(
        np.mean(pct_active),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(pct_active):.1f}%",
    )
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Active words (%)", fontsize=12)
    ax.set_title(
        "Sparsity per dimension (words > 10% max loading)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "SRF produces sparse, interpretable dimensions",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "plots" / "sparsity_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_dimension_coherence_analysis(coherence_scores, sparsity_metrics, output_dir):
    """Plot relationship between sparsity and coherence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Coherence per dimension
    ax = axes[0]
    n_dims = len(coherence_scores)
    colors = plt.cm.viridis(np.linspace(0, 1, n_dims))

    bars = ax.bar(
        range(n_dims), coherence_scores, color=colors, alpha=0.8, edgecolor="black"
    )
    ax.axhline(
        coherence_scores.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {coherence_scores.mean():.4f}",
    )
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Semantic coherence", fontsize=12)
    ax.set_title("Semantic coherence per dimension", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Plot 2: Scatter of Gini vs Coherence
    ax = axes[1]
    ax.scatter(
        sparsity_metrics["gini"],
        coherence_scores,
        s=100,
        c=range(n_dims),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )

    for dim in range(min(n_dims, 10)):
        ax.annotate(
            f"{dim}",
            (sparsity_metrics["gini"].iloc[dim], coherence_scores[dim]),
            fontsize=9,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Sparsity (Gini coefficient)", fontsize=12)
    ax.set_ylabel("Semantic coherence", fontsize=12)
    ax.set_title(
        "High sparsity enables semantic coherence", fontsize=13, fontweight="bold"
    )
    ax.grid(alpha=0.3)

    plt.suptitle(
        "SRF dimension quality metrics",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "plots" / "dimension_quality.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_reconstruction_quality(network, W, observed_mask, output_dir):
    """Visualize reconstruction quality on observed entries."""
    network_clean = np.nan_to_num(network, nan=0)
    reconstruction = W @ W.T
    np.fill_diagonal(reconstruction, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Observed entries - actual vs predicted
    ax = axes[0]
    observed_actual = network_clean[observed_mask]
    observed_pred = reconstruction[observed_mask]

    if len(observed_actual) > 5000:
        idx = np.random.choice(len(observed_actual), 5000, replace=False)
        observed_actual = observed_actual[idx]
        observed_pred = observed_pred[idx]

    ax.scatter(
        observed_actual,
        observed_pred,
        alpha=0.3,
        s=10,
        c="#3498db",
        edgecolors="none",
    )

    max_val = max(observed_actual.max(), observed_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect prediction")

    corr = np.corrcoef(observed_actual, observed_pred)[0, 1]
    mse = np.mean((observed_actual - observed_pred) ** 2)

    ax.set_xlabel("Observed association strength", fontsize=12)
    ax.set_ylabel("SRF reconstruction", fontsize=12)
    ax.set_title(
        f"Reconstruction on observed entries\nr = {corr:.3f}, MSE = {mse:.6f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Reconstruction error distribution
    ax = axes[1]
    errors = observed_actual - observed_pred
    ax.hist(errors, bins=50, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Zero error")
    ax.axvline(
        errors.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {errors.mean():.4f}",
    )

    ax.set_xlabel("Reconstruction error", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of reconstruction errors", fontsize=13, fontweight="bold"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "SRF reconstruction quality with missing data",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "plots" / "reconstruction_quality.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def main(output_dir=None, rank=20, use_all_responses=True, top_n_words=None):
    """
    Run word association analysis pipeline.

    Parameters
    ----------
    output_dir : str, optional
        Output directory for results
    rank : int
        Number of dimensions for SRF
    use_all_responses : bool
        If True, use R123 (all 3 responses). If False, use R1 only.
    top_n_words : int, optional
        If provided, keep only top N words by total outgoing strength
    """
    data_dir = Path("data/small-world-of-words")

    if output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d/%H%M%S")
        output_dir = (
            Path("experiments/development/word_association/outputs") / timestamp
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Load data
    print(f"Loading SWOW data from {data_dir}")

    df_long = load_swow_data_paper_format(data_dir, use_all_responses=use_all_responses)
    print(f"Loaded {len(df_long)} cue-response associations")

    # *** NEW STEP: Calculate PPMI ***
    # This function adds the 'ppmi_strength' column to df_long
    df_long = calculate_ppmi(df_long)

    # Build directed graph
    print("\nBuilding directed graph (using PPMI)")

    # *** MODIFIED CALL ***
    # Tell the graph builder to use the new 'ppmi_strength' column
    G_directed, removed_nodes = build_directed_graph(
        df_long, weight_col="ppmi_strength"
    )

    print(
        f"Graph: {G_directed.number_of_nodes()} nodes, {G_directed.number_of_edges()} edges"
    )
    print(f"Removed {len(removed_nodes)} nodes (not in largest SCC)")

    if top_n_words is not None:
        G_directed = filter_by_node_degree_postgraph(G_directed, top_n_words)

    # Symmetrize with NaN preservation
    print("\nSymmetrizing with NaN preservation")
    # similarity, vocabulary, metadata = symmetrize_vectorized(G_directed)
    similarity, vocabulary, metadata = symmetrize_with_nan_preservation(G_directed)
    print(f"Vocabulary: {len(vocabulary)} words")
    print(f"Missing data: {metadata['pct_missing']:.1f}% (preserved as NaN)")
    print(f"Mean asymmetry: {metadata['mean_asymmetry']:.3f}")

    observed_mask = ~np.isnan(similarity)

    # Fit SRF
    print(f"\nFitting SRF with rank={rank}")
    W_srf = fit_srf(similarity, rank=rank)

    # Compute metrics
    print("\nComputing interpretability metrics")
    coherence_scores = compute_dimension_coherence(W_srf, similarity, top_n=50)
    sparsity_metrics = compute_sparsity(W_srf)
    top_words_df = get_top_words_per_dimension(W_srf, vocabulary, top_n=20)

    print(f"Mean coherence: {coherence_scores.mean():.4f}")
    print(f"Mean sparsity (Gini): {sparsity_metrics['gini'].mean():.3f}")

    # Generate visualizations
    print("\nGenerating visualizations")
    plot_dimension_wordclouds(W_srf, vocabulary, output_dir, n_dims=min(6, rank))

    n_dims_to_plot = min(rank, 10)
    for batch_start in range(0, n_dims_to_plot, 2):
        dims_batch = list(range(batch_start, min(batch_start + 2, n_dims_to_plot)))
        plot_semantic_network(
            similarity, W_srf, vocabulary, output_dir, dims=dims_batch, top_n=25
        )

    plot_sparsity_distribution(W_srf, output_dir)
    plot_dimension_coherence_analysis(coherence_scores, sparsity_metrics, output_dir)
    plot_loading_heatmap(W_srf, vocabulary, output_dir, top_n_per_dim=20)
    plot_reconstruction_quality(similarity, W_srf, observed_mask, output_dir)

    # Save results
    print("\nSaving results")
    np.save(output_dir / "srf_factors.npy", W_srf)
    np.save(output_dir / "similarity_matrix.npy", similarity)

    with open(output_dir / "vocabulary.txt", "w") as f:
        f.write("\n".join(str(word) for word in vocabulary if isinstance(word, str)))

    sparsity_metrics.to_csv(output_dir / "sparsity_metrics.csv", index=False)

    coherence_df = pd.DataFrame(
        {"dimension": range(len(coherence_scores)), "coherence": coherence_scores}
    )
    coherence_df.to_csv(output_dir / "coherence_scores.csv", index=False)

    top_words_df.to_csv(output_dir / "top_words_per_dimension.csv", index=False)

    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(output_dir / "graph_metadata.csv", index=False)

    # Compute reconstruction metrics
    reconstruction = W_srf @ W_srf.T
    np.fill_diagonal(reconstruction, 0)

    sim_actual = similarity[observed_mask]
    sim_pred = reconstruction[observed_mask]
    corr = np.corrcoef(sim_actual, sim_pred)[0, 1]
    mse = np.mean((sim_actual - sim_pred) ** 2)

    metrics_summary = pd.DataFrame(
        [
            {
                "n_words": metadata["n_words"],
                "rank": rank,
                "pct_missing": metadata["pct_missing"],
                "mean_asymmetry": metadata["mean_asymmetry"],
                "mean_gini": sparsity_metrics["gini"].mean(),
                "mean_coherence": coherence_scores.mean(),
                "reconstruction_corr": corr,
                "reconstruction_mse": mse,
            }
        ]
    )
    metrics_summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    print(f"\nResults saved to: {output_dir}")
    print("\nSummary:")
    print(f"  Vocabulary: {len(vocabulary)} words")
    print(f"  Missing data: {metadata['pct_missing']:.1f}%")
    print(f"  Dimensions: {rank}")
    print(f"  Mean sparsity: {sparsity_metrics['gini'].mean():.3f}")
    print(f"  Mean coherence: {coherence_scores.mean():.4f}")
    print(f"  Reconstruction r: {corr:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Word association network analysis with SRF (paper-faithful implementation)"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--rank", type=int, default=20, help="SRF rank/dimensionality")
    parser.add_argument(
        "--use-all-responses",
        action="store_true",
        default=True,
        help="Use R123 (all 3 responses) vs R1 only",
    )
    parser.add_argument(
        "--r1-only",
        action="store_true",
        help="Use only R1 (first response) instead of R123",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=None,
        help="Keep only top N words by total node degree",
    )

    args = parser.parse_args()

    use_all_responses = not args.r1_only

    main(
        output_dir=args.output_dir,
        rank=args.rank,
        use_all_responses=use_all_responses,
        top_n_words=args.top_n_words,
    )
