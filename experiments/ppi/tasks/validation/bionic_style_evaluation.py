"""
BIONIC-style functional evaluation following Nature Methods conventions.

Implements three core tasks:
1. Co-annotation prediction: pairwise functional similarity
2. Module detection: clustering + Jaccard overlap with CORUM complexes
3. Gene function prediction: multilabel classification (uses existing code)

Methods compared:
- SRF (rank-50 embedding)
- Node2Vec (rank-50 embedding)
- STRING network (using CN as pairwise similarity proxy)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from analyses.ppi.corum import load_corum
from analyses.ppi.graph_utils import load_network, build_sparse_adjacency

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 10


def jaccard(set1: set, set2: set) -> float:
    """Compute Jaccard overlap between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def create_coannot_labels(
    proteins: np.ndarray,
    corum_complexes: dict[str, set[str]],
    n_positive: int = 10000,
    n_negative: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create positive/negative pairs for co-annotation prediction.

    Positive pairs: proteins sharing ≥1 CORUM complex
    Negative pairs: proteins sharing no CORUM complex (random sample)
    """
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    n_proteins = len(proteins)

    # Build protein -> complexes mapping
    protein_to_complexes = {i: set() for i in range(n_proteins)}
    for complex_name, members in corum_complexes.items():
        for member in members:
            if member in protein_to_idx:
                idx = protein_to_idx[member]
                protein_to_complexes[idx].add(complex_name)

    # Find positive pairs (share ≥1 complex)
    positive_pairs = []
    for i in range(n_proteins):
        if len(protein_to_complexes[i]) == 0:
            continue
        for j in range(i + 1, n_proteins):
            if len(protein_to_complexes[j]) == 0:
                continue
            shared = protein_to_complexes[i] & protein_to_complexes[j]
            if len(shared) > 0:
                positive_pairs.append((i, j))

    # Sample positives if too many
    if len(positive_pairs) > n_positive:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(positive_pairs), n_positive, replace=False)
        positive_pairs = [positive_pairs[i] for i in idx]

    # Sample negative pairs (share no complex)
    negative_pairs = []
    rng = np.random.RandomState(42)
    proteins_with_complex = [
        i for i in range(n_proteins) if len(protein_to_complexes[i]) > 0
    ]

    max_attempts = n_negative * 10
    attempts = 0
    while len(negative_pairs) < n_negative and attempts < max_attempts:
        i = rng.choice(proteins_with_complex)
        j = rng.choice(proteins_with_complex)
        if i >= j:
            continue
        shared = protein_to_complexes[i] & protein_to_complexes[j]
        if len(shared) == 0:
            negative_pairs.append((i, j))
        attempts += 1

    # Combine and create labels
    all_pairs = positive_pairs + negative_pairs
    labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))
    pairs = np.array(all_pairs)

    return pairs, labels


def evaluate_coannot_prediction(
    embedding: np.ndarray, pairs: np.ndarray, labels: np.ndarray, method: str = "cosine"
) -> dict:
    """Evaluate co-annotation prediction using embedding similarity."""
    # Compute pairwise similarities
    if method == "cosine":
        sim_matrix = cosine_similarity(embedding)
        scores = sim_matrix[pairs[:, 0], pairs[:, 1]]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    return {"auroc": auroc, "aupr": aupr, "scores": scores}


def evaluate_coannot_cn(
    adj_sparse: csr_matrix,
    pairs: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Evaluate co-annotation using Common Neighbors baseline."""
    # Compute CN similarity matrix
    cn_matrix = adj_sparse @ adj_sparse

    # Get scores for pairs
    scores = np.array([cn_matrix[i, j] for i, j in pairs]).flatten()

    # Compute metrics
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    return {"auroc": auroc, "aupr": aupr, "scores": scores}


def cluster_embedding_knn_louvain(
    embedding: np.ndarray,
    k_neighbors: int = 20,
) -> np.ndarray:
    """Cluster embedding using Louvain on k-NN graph."""
    # Build k-NN graph in embedding space
    knn_graph = kneighbors_graph(
        embedding, k_neighbors, mode="connectivity", include_self=False
    )

    # Convert to networkx graph
    g = nx.from_scipy_sparse_array(knn_graph)

    # Run Louvain clustering
    try:
        communities = nx.community.louvain_communities(g, seed=42)
    except AttributeError:
        # Fallback for older networkx versions
        import networkx.algorithms.community as nx_comm

        communities = nx_comm.louvain_communities(g, seed=42)

    # Convert to labels array
    labels = np.zeros(len(embedding), dtype=int)
    for cluster_idx, community in enumerate(communities):
        for node_idx in community:
            labels[node_idx] = cluster_idx

    return labels


def evaluate_module_detection(
    embedding: np.ndarray,
    proteins: np.ndarray,
    corum_complexes: dict[str, set[str]],
    k_neighbors_values: list[int] = [10, 20, 30, 50],
    min_complex_size: int = 3,
    max_complex_size: int = 50,
) -> dict:
    """Evaluate module detection via Louvain clustering on k-NN graph + Jaccard overlap with CORUM.

    Returns best results across k_neighbors values.
    """
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    # Filter complexes by size
    valid_complexes = {}
    for name, members in corum_complexes.items():
        present = [m for m in members if m in protein_to_idx]
        if min_complex_size <= len(present) <= max_complex_size:
            valid_complexes[name] = set([protein_to_idx[m] for m in present])

    best_mean_jaccard = 0.0
    best_results = None

    # Try different k_neighbors values for k-NN graph construction
    for k_neighbors in k_neighbors_values:
        cluster_labels = cluster_embedding_knn_louvain(
            embedding, k_neighbors=k_neighbors
        )

        # Build cluster -> proteins mapping
        n_clusters = cluster_labels.max() + 1
        clusters = {i: set() for i in range(n_clusters)}
        for protein_idx, cluster_idx in enumerate(cluster_labels):
            clusters[cluster_idx].add(protein_idx)

        # Compute Jaccard for each complex
        jaccard_scores = []
        complex_names = []

        for complex_name, complex_members in valid_complexes.items():
            # Find best matching cluster
            best_jaccard = 0.0
            for cluster_idx, cluster_members in clusters.items():
                j = jaccard(complex_members, cluster_members)
                if j > best_jaccard:
                    best_jaccard = j

            jaccard_scores.append(best_jaccard)
            complex_names.append(complex_name)

        # Compute summary metrics
        jaccard_scores = np.array(jaccard_scores)
        mean_jaccard = jaccard_scores.mean()
        n_captured = (jaccard_scores >= 0.5).sum()

        if mean_jaccard > best_mean_jaccard:
            best_mean_jaccard = mean_jaccard
            best_results = {
                "k_neighbors": k_neighbors,
                "n_clusters": n_clusters,
                "mean_jaccard": mean_jaccard,
                "n_captured": n_captured,
                "jaccard_scores": jaccard_scores,
                "complex_names": complex_names,
                "n_complexes": len(valid_complexes),
            }

    return best_results


def plot_coannot_bars(results: dict, output_dir: Path):
    """Plot co-annotation prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = list(results.keys())
    aurocs = [results[m]["auroc"] for m in methods]
    auprs = [results[m]["aupr"] for m in methods]

    colors = {"SRF": "#2E86AB", "Node2Vec": "#A23B72", "STRING_CN": "#999999"}
    bar_colors = [colors.get(m, "#999999") for m in methods]

    # AUROC
    ax = axes[0]
    ax.bar(methods, aurocs, color=bar_colors, alpha=0.7, width=0.6)
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_title("Co-annotation prediction (AUROC)", fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    # AUPR
    ax = axes[1]
    ax.bar(methods, auprs, color=bar_colors, alpha=0.7, width=0.6)
    ax.set_ylabel("AUPR", fontweight="bold")
    ax.set_title("Co-annotation prediction (AUPR)", fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "coannot_bars.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "coannot_bars.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_module_detection_bars(results: dict, output_dir: Path):
    """Plot module detection results."""
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = list(results.keys())
    mean_jaccards = [results[m]["mean_jaccard"] for m in methods]
    n_captured = [results[m]["n_captured"] for m in methods]

    colors = {"SRF": "#2E86AB", "Node2Vec": "#A23B72", "STRING_CN": "#999999"}
    bar_colors = [colors.get(m, "#999999") for m in methods]

    bars = ax.bar(methods, mean_jaccards, color=bar_colors, alpha=0.7, width=0.6)

    # Add captured count above bars
    for i, (bar, n) in enumerate(zip(bars, n_captured)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{n}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Mean Jaccard overlap", fontweight="bold")
    ax.set_title("Module detection (captured complexes above bars)", fontweight="bold")
    ax.set_ylim([0, max(mean_jaccards) * 1.2])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "module_detection_bars.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "module_detection_bars.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_jaccard_scatter(
    srf_results: dict, baseline_results: dict, baseline_name: str, output_dir: Path
):
    """Plot per-complex Jaccard scatter (SRF vs baseline)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    srf_jaccards = srf_results["jaccard_scores"]
    baseline_jaccards = baseline_results["jaccard_scores"]

    ax.scatter(baseline_jaccards, srf_jaccards, alpha=0.5, s=20, color="#2E86AB")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)

    ax.set_xlabel(f"{baseline_name} Jaccard", fontweight="bold")
    ax.set_ylabel("SRF Jaccard", fontweight="bold")
    ax.set_title("Per-complex Jaccard overlap", fontweight="bold")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)

    # Count points above/below diagonal
    above = (srf_jaccards > baseline_jaccards).sum()
    below = (srf_jaccards < baseline_jaccards).sum()
    ax.text(
        0.05,
        0.95,
        f"SRF better: {above}\n{baseline_name} better: {below}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"jaccard_scatter_vs_{baseline_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        output_dir / f"jaccard_scatter_vs_{baseline_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_jaccard_histograms(results: dict, output_dir: Path):
    """Plot distribution of Jaccard scores per method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"SRF": "#2E86AB", "Node2Vec": "#A23B72", "STRING_CN": "#999999"}

    for method_name in results.keys():
        jaccards = results[method_name]["jaccard_scores"]
        color = colors.get(method_name, "#999999")

        ax.hist(
            jaccards, bins=30, alpha=0.5, label=method_name, color=color, density=True
        )
        mean_j = jaccards.mean()
        ax.axvline(mean_j, color=color, linestyle="--", lw=2, alpha=0.8)

    ax.set_xlabel("Jaccard overlap", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Distribution of per-complex Jaccard scores", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "jaccard_histograms.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "jaccard_histograms.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    srf_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    baseline_dir = base_dir / "experiments" / "ppi" / "validation" / "baselines"
    output_dir = (
        base_dir
        / "experiments"
        / "ppi"
        / "validation"
        / "outputs"
        / "bionic_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    srf_embedding = np.load(srf_dir / "embedding.npy")
    node2vec_embedding = np.load(baseline_dir / "node2vec_embedding.npy")

    with open(srf_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])

    # Load CORUM complexes
    corum_path = data_dir / "corum_complexes.txt"
    corum_complexes = load_corum(corum_path)

    # Load STRING network
    string_path = data_dir / "STRING_human_min900_v12.csv"
    g, _ = load_network(string_path)
    adj_sparse = build_sparse_adjacency(g, sorted(g.nodes()))

    # === CO-ANNOTATION PREDICTION ===
    pairs, labels = create_coannot_labels(
        proteins, corum_complexes, n_positive=5000, n_negative=5000
    )

    coannot_results = {}
    coannot_results["SRF"] = evaluate_coannot_prediction(srf_embedding, pairs, labels)
    coannot_results["Node2Vec"] = evaluate_coannot_prediction(
        node2vec_embedding, pairs, labels
    )
    coannot_results["STRING_CN"] = evaluate_coannot_cn(adj_sparse, pairs, labels)

    # === MODULE DETECTION ===
    module_results = {}
    module_results["SRF"] = evaluate_module_detection(
        srf_embedding, proteins, corum_complexes, k_neighbors_values=[10, 20, 30, 50]
    )
    module_results["Node2Vec"] = evaluate_module_detection(
        node2vec_embedding,
        proteins,
        corum_complexes,
        k_neighbors_values=[10, 20, 30, 50],
    )

    # For STRING, use Louvain clustering on the graph instead of k-means
    # For simplicity, we'll skip STRING module detection for now
    # (BIONIC paper uses network-based clustering for input networks)

    # === PLOTS ===
    plot_coannot_bars(coannot_results, output_dir)
    plot_module_detection_bars(module_results, output_dir)
    plot_jaccard_scatter(
        module_results["SRF"], module_results["Node2Vec"], "Node2Vec", output_dir
    )
    plot_jaccard_histograms(module_results, output_dir)

    # === SAVE RESULTS ===
    results_summary = []

    # Co-annotation
    for method in coannot_results.keys():
        results_summary.append(
            {
                "task": "co-annotation",
                "method": method,
                "metric": "AUROC",
                "value": coannot_results[method]["auroc"],
            }
        )
        results_summary.append(
            {
                "task": "co-annotation",
                "method": method,
                "metric": "AUPR",
                "value": coannot_results[method]["aupr"],
            }
        )

    # Module detection
    for method in module_results.keys():
        results_summary.append(
            {
                "task": "module_detection",
                "method": method,
                "metric": "mean_jaccard",
                "value": module_results[method]["mean_jaccard"],
            }
        )
        results_summary.append(
            {
                "task": "module_detection",
                "method": method,
                "metric": "n_captured",
                "value": module_results[method]["n_captured"],
            }
        )
        results_summary.append(
            {
                "task": "module_detection",
                "method": method,
                "metric": "k_neighbors",
                "value": module_results[method]["k_neighbors"],
            }
        )
        results_summary.append(
            {
                "task": "module_detection",
                "method": method,
                "metric": "n_clusters",
                "value": module_results[method]["n_clusters"],
            }
        )

    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(output_dir / "bionic_evaluation_results.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("BIONIC-STYLE FUNCTIONAL EVALUATION")
    print("=" * 60)

    print("\nCo-annotation Prediction:")
    for method in coannot_results.keys():
        auroc = coannot_results[method]["auroc"]
        aupr = coannot_results[method]["aupr"]
        print(f"  {method:15s} AUROC: {auroc:.3f}  AUPR: {aupr:.3f}")

    print("\nModule Detection:")
    for method in module_results.keys():
        mean_j = module_results[method]["mean_jaccard"]
        n_cap = module_results[method]["n_captured"]
        k_neighbors = module_results[method]["k_neighbors"]
        n_clusters = module_results[method]["n_clusters"]
        n_total = module_results[method]["n_complexes"]
        print(
            f"  {method:15s} Mean J: {mean_j:.3f}  Captured: {n_cap}/{n_total}  (k_neighbors={k_neighbors}, n_clusters={n_clusters})"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
