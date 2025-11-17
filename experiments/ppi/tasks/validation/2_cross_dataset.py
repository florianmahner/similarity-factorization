import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from analyses.ppi.graph_utils import load_network, build_sparse_adjacency
from analyses.ppi.corum import map_string_ids_to_genes
from analyses.ppi.metrics import compute_link_prediction_metrics


def map_ensg_to_gene_names(ensg_ids: list[str]) -> dict[str, str]:
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            ensg_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            returnall=True,
        )

        ensg_to_gene = {}
        for result in results["out"]:
            if "symbol" in result and "query" in result:
                ensg_to_gene[result["query"]] = result["symbol"]

        return ensg_to_gene
    except ImportError:
        print("Warning: mygene not available, trying pandas-based mapping")
        return {}


def load_huri_network(csv_path: Path, gene_mapping: dict[str, str]) -> tuple[set, list]:
    df = pd.read_csv(csv_path)

    edges = set()
    proteins = set()

    for _, row in df.iterrows():
        u_ensg = row["source"]
        v_ensg = row["target"]

        u_gene = gene_mapping.get(u_ensg, u_ensg)
        v_gene = gene_mapping.get(v_ensg, v_ensg)

        u, v = sorted([u_gene, v_gene])
        edges.add((u, v))
        proteins.add(u)
        proteins.add(v)

    return edges, sorted(proteins)


def find_novel_huri_edges(huri_edges: set, string_edges: set) -> list:
    novel = huri_edges - string_edges
    return list(novel)


def evaluate_embedding_on_edges(
    embedding: np.ndarray,
    proteins: np.ndarray,
    positive_edges: list,
    negative_edges: list,
) -> dict:
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    # Filter edges to only include proteins in embedding
    pos_edges_filtered = [
        (u, v) for u, v in positive_edges if u in protein_to_idx and v in protein_to_idx
    ]
    neg_edges_filtered = [
        (u, v) for u, v in negative_edges if u in protein_to_idx and v in protein_to_idx
    ]

    print(
        f"Positive edges in embedding: {len(pos_edges_filtered)}/{len(positive_edges)}"
    )
    print(
        f"Negative edges in embedding: {len(neg_edges_filtered)}/{len(negative_edges)}"
    )

    # Compute scores for positive edges
    pos_scores = []
    for u, v in pos_edges_filtered:
        i, j = protein_to_idx[u], protein_to_idx[v]
        score = np.dot(embedding[i], embedding[j])
        pos_scores.append(score)

    # Compute scores for negative edges
    neg_scores = []
    for u, v in neg_edges_filtered:
        i, j = protein_to_idx[u], protein_to_idx[v]
        score = np.dot(embedding[i], embedding[j])
        neg_scores.append(score)

    # Combine scores and labels
    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

    # Compute metrics
    metrics = compute_link_prediction_metrics(scores, labels)

    return metrics


def evaluate_cn_on_edges(
    adj_sparse: csr_matrix,
    proteins: np.ndarray,
    positive_edges: list,
    negative_edges: list,
) -> dict:
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    # Compute CN similarity matrix
    cn_matrix = adj_sparse @ adj_sparse

    # Filter edges
    pos_edges_filtered = [
        (u, v) for u, v in positive_edges if u in protein_to_idx and v in protein_to_idx
    ]
    neg_edges_filtered = [
        (u, v) for u, v in negative_edges if u in protein_to_idx and v in protein_to_idx
    ]

    # Compute scores
    pos_scores = []
    for u, v in pos_edges_filtered:
        i, j = protein_to_idx[u], protein_to_idx[v]
        score = cn_matrix[i, j]
        pos_scores.append(score)

    neg_scores = []
    for u, v in neg_edges_filtered:
        i, j = protein_to_idx[u], protein_to_idx[v]
        score = cn_matrix[i, j]
        neg_scores.append(score)

    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

    metrics = compute_link_prediction_metrics(scores, labels)

    return metrics


def sample_negative_edges(
    all_proteins: set, positive_edges: set, n_samples: int, seed: int = 42
) -> list:
    """Sample negative edges (non-edges) from the protein space.

    Parameters
    ----------
    all_proteins : set
        Set of all proteins
    positive_edges : set
        Set of positive edges to exclude
    n_samples : int
        Number of negative samples
    seed : int
        Random seed

    Returns
    -------
    list
        List of negative edge tuples
    """
    np.random.seed(seed)
    proteins_list = sorted(all_proteins)
    n_proteins = len(proteins_list)

    negative_edges = []
    attempts = 0
    max_attempts = n_samples * 100

    while len(negative_edges) < n_samples and attempts < max_attempts:
        i = np.random.randint(0, n_proteins)
        j = np.random.randint(0, n_proteins)

        if i == j:
            continue

        u, v = sorted([proteins_list[i], proteins_list[j]])
        edge = (u, v)

        if edge not in positive_edges and edge not in negative_edges:
            negative_edges.append(edge)

        attempts += 1

    return negative_edges


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar plot of metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["auroc", "auprc"]
    titles = ["AUROC", "AUPRC"]

    for ax, metric, title in zip(axes, metrics, titles):
        srf_val = results_df.loc[results_df["method"] == "SRF", metric].values[0]
        cn_val = results_df.loc[results_df["method"] == "CN", metric].values[0]

        methods = ["SRF (Rank-50)", "CN Baseline"]
        values = [srf_val, cn_val]

        ax.bar(methods, values, alpha=0.7, color=["#2E86AB", "#A23B72"])
        ax.set_ylabel(title)
        ax.set_title(f"Cross-Dataset Validation: {title}")
        ax.set_ylim([0, max(values) * 1.2])
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(
                i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_dir / "cross_dataset_metrics.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Detailed metrics table plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    for _, row in results_df.iterrows():
        table_data.append(
            [
                row["method"],
                f"{row['auroc']:.3f}",
                f"{row['auprc']:.4f}",
                f"{row['p500']:.4f}",
                f"{row['ndcg']:.3f}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "AUROC", "AUPRC", "P@500", "NDCG"],
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.savefig(output_dir / "metrics_table.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Paths
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    validation_output_dir = (
        base_dir / "experiments" / "ppi" / "validation" / "outputs" / "cross_dataset"
    )

    # Load SRF embedding (trained on STRING)
    print("Loading rank-50 SRF embedding (trained on STRING)...")
    embedding = np.load(output_dir / "embedding.npy")
    print(f"Embedding shape: {embedding.shape}")

    # Load protein names
    with open(output_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])
    print(f"Number of proteins: {len(proteins)}")

    # Load STRING network
    print("\nLoading STRING network...")
    string_path = data_dir / "STRING_human_min900_v12.csv"
    g_string, _ = load_network(string_path)

    # Get STRING edges as set
    string_nodes = sorted(g_string.nodes())
    mapping_file = data_dir / "protein_info_900.csv"
    if mapping_file.exists():
        string_nodes, _ = map_string_ids_to_genes(string_nodes, mapping_file)

    string_edges = set()
    for u, v in g_string.edges():
        u_sorted, v_sorted = sorted([u, v])
        string_edges.add((u_sorted, v_sorted))

    print(f"STRING edges: {len(string_edges)}")
    print(f"STRING nodes: {len(string_nodes)}")

    # Build STRING adjacency matrix for baselines
    adj_sparse = build_sparse_adjacency(g_string, string_nodes)

    print("\nMapping HuRI ENSG IDs to gene names...")
    huri_df = pd.read_csv(data_dir / "HuRI.csv")
    all_ensg = list(set(huri_df["source"].tolist() + huri_df["target"].tolist()))
    ensg_to_gene = map_ensg_to_gene_names(all_ensg)
    print(f"Mapped {len(ensg_to_gene)}/{len(all_ensg)} ENSG IDs")

    print("\nLoading HuRI network...")
    huri_path = data_dir / "HuRI.csv"
    huri_edges, huri_proteins = load_huri_network(huri_path, ensg_to_gene)
    print(f"HuRI edges: {len(huri_edges)}")
    print(f"HuRI proteins: {len(huri_proteins)}")

    # Find novel HuRI edges (not in STRING)
    print("\nFinding novel HuRI edges...")
    novel_edges = find_novel_huri_edges(huri_edges, string_edges)
    print(f"Novel HuRI edges (not in STRING): {len(novel_edges)}")

    overlap_edges = huri_edges & string_edges
    print(f"Overlapping edges (in both): {len(overlap_edges)}")

    # Sample negative edges
    print("\nSampling negative edges...")
    all_proteins = set(huri_proteins) | set(string_nodes)
    all_edges = huri_edges | string_edges
    n_negatives = len(novel_edges)  # Balance positive and negative
    negative_edges = sample_negative_edges(
        all_proteins, all_edges, n_negatives, seed=42
    )
    print(f"Negative edges sampled: {len(negative_edges)}")

    # Evaluate SRF
    print("\n" + "=" * 50)
    print("Evaluating SRF (Rank-50)")
    print("=" * 50)
    srf_metrics = evaluate_embedding_on_edges(
        embedding, proteins, novel_edges, negative_edges
    )

    # Evaluate CN baseline
    print("\n" + "=" * 50)
    print("Evaluating Common Neighbors Baseline")
    print("=" * 50)
    cn_metrics = evaluate_cn_on_edges(adj_sparse, proteins, novel_edges, negative_edges)

    # Compile results
    results = []
    results.append({"method": "SRF", **srf_metrics})
    results.append({"method": "CN", **cn_metrics})
    results_df = pd.DataFrame(results)

    # Save results
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(validation_output_dir / "results.csv", index=False)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print("\nCross-Dataset Validation: STRING → HuRI (Novel Edges)")
    print(results_df.to_string(index=False))

    # Compute improvements
    auroc_improvement = (
        (srf_metrics["auroc"] - cn_metrics["auroc"]) / cn_metrics["auroc"] * 100
    )
    auprc_improvement = (
        (srf_metrics["auprc"] - cn_metrics["auprc"]) / cn_metrics["auprc"] * 100
    )

    print(f"\nSRF Improvement over CN:")
    print(f"  AUROC: +{auroc_improvement:.1f}%")
    print(f"  AUPRC: +{auprc_improvement:.1f}%")

    # Create plots
    print("\nGenerating plots...")
    plot_results(results_df, validation_output_dir)

    print(f"\nResults saved to: {validation_output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
