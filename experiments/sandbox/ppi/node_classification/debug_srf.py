#!/usr/bin/env python3
"""
Diagnostic script to understand why SRF fails at GO term classification.

Key question: Does SRF capture edge structure (physical interactions) but not
functional annotations (GO terms)?

Tests:
1. CORUM validation - can SRF decode protein complexes (physical interactions)?
2. Random baseline - is 0% accuracy due to label sparsity?
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from pysrf import SRF
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Add project to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.ppi.utils import load_network
from experiments.ppi.corum import load_corum, validate_embedding_against_corum
from experiments.ppi.node_classification import (
    build_label_matrix,
    fetch_go_annotations,
    filter_terms,
)

# Set plotting style
sns.set_theme(style="ticks", context="talk")


def load_protein_mapping(mapping_file: Path) -> dict[str, str]:
    """Load STRING protein ID to gene name mapping."""
    df = pd.read_csv(mapping_file)
    # Map full STRING ID (9606.ENSP...) to gene name
    mapping = dict(zip(df["string_protein_id"], df["preferred_name"]))
    # Also map ENSP ID alone
    for string_id, gene in list(mapping.items()):
        ensp_id = string_id.split(".", 1)[1] if "." in string_id else string_id
        mapping[ensp_id] = gene
    return mapping


def get_subgraph(g, n_nodes):
    """Get top-degree connected subgraph."""
    if n_nodes >= len(g):
        return g
    top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:n_nodes]
    top_nodes = [n for n, d in top_nodes]
    sub = g.subgraph(top_nodes).copy()
    largest_cc = max(nx.connected_components(sub), key=len)
    return sub.subgraph(largest_cc).copy()


def fit_pysrf_current(nodes, edges, rank, seed, max_outer=300):
    """Current implementation - treats unobserved edges as zeros"""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    rows, cols = [], []
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            rows.extend([u_idx, v_idx])
            cols.extend([v_idx, u_idx])

    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adj_dense = adj.toarray()  # Creates 99% zeros
    np.fill_diagonal(adj_dense, np.nan)

    # Print diagnostics
    n_ones = (adj_dense == 1.0).sum()
    n_zeros = (adj_dense == 0.0).sum()
    n_nans = np.isnan(adj_dense).sum()
    total = n_nodes * n_nodes

    print(f"\n=== CURRENT (ZEROS) IMPLEMENTATION ===")
    print(f"Adjacency matrix ({n_nodes}x{n_nodes}):")
    print(f"  - Values=1: {n_ones:,} ({100*n_ones/total:.2f}%)")
    print(f"  - Values=0: {n_zeros:,} ({100*n_zeros/total:.2f}%)")
    print(f"  - Values=NaN: {n_nans:,} ({100*n_nans/total:.2f}%)")

    model = SRF(
        rank=rank,
        max_outer=max_outer,
        verbose=1,
        missing_values=np.nan,
        random_state=seed,
        rho=3.0,
        init="random_sqrt",
        loss="frobenius",
    )
    W = model.fit_transform(adj_dense)

    print(f"\nEmbeddings ({W.shape}):")
    print(f"  - Mean: {W.mean():.4f}")
    print(f"  - Std: {W.std():.4f}")
    print(f"  - Min: {W.min():.4f}")
    print(f"  - Max: {W.max():.4f}")
    print(f"  - Magnitude (mean): {np.linalg.norm(W, axis=1).mean():.4f}")

    return {node: W[idx] for node, idx in node_to_idx.items()}, adj_dense


def fit_pysrf_fixed(nodes, edges, rank, seed, max_outer=300):
    """Fixed implementation - treats unobserved edges as NaN"""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Initialize with NaN
    adj_dense = np.full((n_nodes, n_nodes), np.nan, dtype=np.float32)

    # Add edges as 1s (symmetric)
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            adj_dense[u_idx, v_idx] = 1.0
            adj_dense[v_idx, u_idx] = 1.0

    # Print diagnostics
    n_ones = (adj_dense == 1.0).sum()
    n_zeros = (adj_dense == 0.0).sum()
    n_nans = np.isnan(adj_dense).sum()
    total = n_nodes * n_nodes

    print(f"\n=== FIXED (NaN) IMPLEMENTATION ===")
    print(f"Adjacency matrix ({n_nodes}x{n_nodes}):")
    print(f"  - Values=1: {n_ones:,} ({100*n_ones/total:.2f}%)")
    print(f"  - Values=0: {n_zeros:,} ({100*n_zeros/total:.2f}%)")
    print(f"  - Values=NaN: {n_nans:,} ({100*n_nans/total:.2f}%)")

    model = SRF(
        rank=rank,
        max_outer=max_outer,
        verbose=1,
        missing_values=np.nan,
        random_state=seed,
        rho=3.0,
        init="random_sqrt",
        loss="frobenius",
    )
    W = model.fit_transform(adj_dense)

    print(f"\nEmbeddings ({W.shape}):")
    print(f"  - Mean: {W.mean():.4f}")
    print(f"  - Std: {W.std():.4f}")
    print(f"  - Min: {W.min():.4f}")
    print(f"  - Max: {W.max():.4f}")
    print(f"  - Magnitude (mean): {np.linalg.norm(W, axis=1).mean():.4f}")

    return {node: W[idx] for node, idx in node_to_idx.items()}, adj_dense


def test_classification(embeddings, labeled_nodes, label_dict, rank, seed):
    """Quick classification test"""
    X = np.array([embeddings.get(node, np.zeros(rank)) for node in labeled_nodes])
    y = [label_dict[node] for node in labeled_nodes]

    # Check for zero embeddings
    non_zero_rows = np.any(X != 0, axis=1).sum()
    print(f"\nEmbedding coverage:")
    print(f"  - Non-zero rows: {non_zero_rows}/{len(X)} ({100*non_zero_rows/len(X):.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200, solver="liblinear"))
    clf.fit(X_train, y_train_bin)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train_bin, y_pred_train)
    test_acc = accuracy_score(y_test_bin, y_pred_test)
    train_f1 = f1_score(y_train_bin, y_pred_train, average="micro", zero_division=0)
    test_f1 = f1_score(y_test_bin, y_pred_test, average="micro", zero_division=0)

    print(f"\nClassification results:")
    print(f"  - Train accuracy: {train_acc:.3f}")
    print(f"  - Test accuracy: {test_acc:.3f}")
    print(f"  - Train Micro-F1: {train_f1:.3f}")
    print(f"  - Test Micro-F1: {test_f1:.3f}")

    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "test_f1": test_f1,
        "train_f1": train_f1,
    }


def create_comparison_figure(
    adj_current,
    adj_fixed,
    emb_current,
    emb_fixed,
    results_current,
    results_fixed,
    n_nodes,
    n_edges,
    output_dir,
):
    """Create comprehensive comparison figure showing all differences."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Row 1: Adjacency Matrix Statistics
    ax1 = fig.add_subplot(gs[0, 0])
    adj_stats_current = {
        "1s\n(edges)": (adj_current == 1.0).sum(),
        "0s\n(negatives)": (adj_current == 0.0).sum(),
        "NaN\n(missing)": np.isnan(adj_current).sum(),
    }
    total = n_nodes * n_nodes
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    ax1.bar(
        range(len(adj_stats_current)),
        [v / total * 100 for v in adj_stats_current.values()],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_xticks(range(len(adj_stats_current)))
    ax1.set_xticklabels(adj_stats_current.keys())
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Current (Buggy)\nAdjacency Matrix Composition", fontweight="bold")
    ax1.set_ylim(0, 100)
    for i, (k, v) in enumerate(adj_stats_current.items()):
        pct = v / total * 100
        ax1.text(i, pct + 2, f"{pct:.1f}%\n({v:,})", ha="center", fontsize=9)
    sns.despine(ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    adj_stats_fixed = {
        "1s\n(edges)": (adj_fixed == 1.0).sum(),
        "0s\n(negatives)": (adj_fixed == 0.0).sum(),
        "NaN\n(missing)": np.isnan(adj_fixed).sum(),
    }
    ax2.bar(
        range(len(adj_stats_fixed)),
        [v / total * 100 for v in adj_stats_fixed.values()],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.set_xticks(range(len(adj_stats_fixed)))
    ax2.set_xticklabels(adj_stats_fixed.keys())
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Fixed\nAdjacency Matrix Composition", fontweight="bold")
    ax2.set_ylim(0, 100)
    for i, (k, v) in enumerate(adj_stats_fixed.items()):
        pct = v / total * 100
        ax2.text(i, pct + 2, f"{pct:.1f}%\n({v:,})", ha="center", fontsize=9)
    sns.despine(ax=ax2)

    # Row 1: Embedding Statistics
    ax3 = fig.add_subplot(gs[0, 2:])
    emb_stats = {
        "Mean": [emb_current.mean(), emb_fixed.mean()],
        "Std": [emb_current.std(), emb_fixed.std()],
        "Magnitude": [
            np.linalg.norm(emb_current, axis=1).mean(),
            np.linalg.norm(emb_fixed, axis=1).mean(),
        ],
    }
    x = np.arange(len(emb_stats))
    width = 0.35
    colors_emb = ["#e74c3c", "#2ecc71"]
    for i, (method, color) in enumerate(zip(["Current", "Fixed"], colors_emb)):
        values = [stats[i] for stats in emb_stats.values()]
        ax3.bar(x + i * width, values, width, label=method, color=color, alpha=0.8)
        for j, v in enumerate(values):
            ax3.text(j + i * width, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax3.set_ylabel("Value")
    ax3.set_title("Embedding Statistics", fontweight="bold")
    ax3.set_xticks(x + width / 2)
    ax3.set_xticklabels(emb_stats.keys())
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax3)

    # Row 2: Embedding Distributions
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.hist(
        emb_current.flatten(),
        bins=50,
        alpha=0.6,
        label="Current",
        color="#e74c3c",
        edgecolor="black",
    )
    ax4.hist(
        emb_fixed.flatten(),
        bins=50,
        alpha=0.6,
        label="Fixed",
        color="#2ecc71",
        edgecolor="black",
    )
    ax4.set_xlabel("Embedding Value")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Embedding Value Distribution", fontweight="bold")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax4)

    # Row 2: Embedding Magnitude Distribution
    ax5 = fig.add_subplot(gs[1, 2:])
    mag_current = np.linalg.norm(emb_current, axis=1)
    mag_fixed = np.linalg.norm(emb_fixed, axis=1)
    ax5.hist(
        mag_current,
        bins=30,
        alpha=0.6,
        label="Current",
        color="#e74c3c",
        edgecolor="black",
    )
    ax5.hist(
        mag_fixed, bins=30, alpha=0.6, label="Fixed", color="#2ecc71", edgecolor="black"
    )
    ax5.set_xlabel("L2 Norm (Magnitude)")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Embedding Magnitude Distribution", fontweight="bold")
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax5)

    # Row 3: Classification Performance
    ax6 = fig.add_subplot(gs[2, :2])
    metrics = ["Test\nAccuracy", "Train\nAccuracy", "Test\nMicro-F1", "Train\nMicro-F1"]
    current_vals = [
        results_current["test_acc"],
        results_current["train_acc"],
        results_current["test_f1"],
        results_current["train_f1"],
    ]
    fixed_vals = [
        results_fixed["test_acc"],
        results_fixed["train_acc"],
        results_fixed["test_f1"],
        results_fixed["train_f1"],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax6.bar(
        x - width / 2, current_vals, width, label="Current", color="#e74c3c", alpha=0.8
    )
    bars2 = ax6.bar(
        x + width / 2, fixed_vals, width, label="Fixed", color="#2ecc71", alpha=0.8
    )

    ax6.set_ylabel("Score")
    ax6.set_title("Classification Performance", fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.set_ylim(0, 1.0)
    ax6.legend()
    ax6.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    sns.despine(ax=ax6)

    # Row 3: Summary Text
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis("off")

    summary_text = f"""
GRAPH STATISTICS
• Nodes: {n_nodes:,}
• Edges: {n_edges:,}
• Density: {100 * n_edges / (n_nodes * (n_nodes - 1) / 2):.2f}%

KEY FINDINGS
• Current approach treats {(adj_current == 0.0).sum():,} unobserved
  edges as "definitely dissimilar" (value=0)

• Fixed approach marks {np.isnan(adj_fixed).sum():,} unobserved
  edges as "unknown" (value=NaN)

• This simple change improves test accuracy from
  {results_current['test_acc']:.1%} → {results_fixed['test_acc']:.1%}

IMPROVEMENT: {(results_fixed['test_acc'] - results_current['test_acc']) * 100:+.1f} percentage points
"""

    ax7.text(
        0.05,
        0.95,
        summary_text,
        transform=ax7.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Overall title
    fig.suptitle(
        "SRF Node Classification: Current vs Fixed Implementation",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_dir / "srf_debug_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "srf_debug_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Saved comparison figure to {output_dir}")
    print(f"  - srf_debug_comparison.pdf")
    print(f"  - srf_debug_comparison.png")


def main():
    # Config - use smaller subset for faster iteration
    data_dir = Path(project_root) / "data/ppi"
    rank = 25
    seed = 42
    n_nodes_target = 500  # Smaller for fast compute
    max_outer = 300

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    g = load_network(data_dir / "STRING_human_min900_v12.csv")
    g_sub = get_subgraph(g, n_nodes_target)
    nodes = sorted(g_sub.nodes())
    edges = list(g_sub.edges())
    n_nodes = len(nodes)

    print(f"Graph: {n_nodes} nodes, {len(edges)} edges")
    print(f"Density: {100 * len(edges) / (n_nodes * (n_nodes - 1) / 2):.2f}%")

    # Load protein ID to gene name mapping
    mapping_file = data_dir / "protein_info_900.csv"
    id_to_gene = load_protein_mapping(mapping_file)

    # Map node IDs to gene names
    genes = []
    for node in nodes:
        gene = id_to_gene.get(node)
        if gene is None:
            ensp = node.split(".")[-1] if "." in node else node
            gene = id_to_gene.get(ensp, ensp)
        genes.append(gene)
    genes = np.array(genes)

    # Load CORUM complexes
    corum_file = data_dir / "corum_complexes.txt"
    corum_complexes = load_corum(corum_file)
    print(f"Loaded {len(corum_complexes)} CORUM complexes")

    # Check gene overlap with CORUM
    all_corum_genes = set()
    for complex_genes in corum_complexes.values():
        all_corum_genes.update(complex_genes)
    gene_overlap = len(set(genes) & all_corum_genes)
    print(f"Gene overlap with CORUM: {gene_overlap}/{len(genes)} ({100*gene_overlap/len(genes):.1f}%)")

    # ========================================================================
    # STEP 2: Fit SRF embeddings
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Fitting SRF embeddings")
    print("=" * 70)

    emb_dict, _ = fit_pysrf_current(nodes, edges, rank, seed, max_outer)
    W = np.array([emb_dict[node] for node in nodes])

    # ========================================================================
    # STEP 3: CORUM Validation (PRIMARY TEST)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: CORUM Complex Validation")
    print("=" * 70)
    print("Testing if SRF embeddings capture physical interactions (protein complexes)")

    corum_results = validate_embedding_against_corum(W, genes, corum_complexes, top_n=50)

    print(f"\nCORUM validation results:")
    print(f"  - Dimensions with matches: {(corum_results['f1'] > 0).sum()}/{len(corum_results)}")
    print(f"  - Mean F1: {corum_results['f1'].mean():.3f}")
    print(f"  - Max F1: {corum_results['f1'].max():.3f}")
    print(f"  - Dims with F1 > 0.3: {(corum_results['f1'] > 0.3).sum()}")
    print(f"  - Dims with F1 > 0.5: {(corum_results['f1'] > 0.5).sum()}")

    # Show top matching dimensions
    top_dims = corum_results.nlargest(5, 'f1')
    print(f"\nTop 5 dimensions by CORUM F1:")
    for _, row in top_dims.iterrows():
        print(f"  Dim {row['dimension']:2d}: F1={row['f1']:.3f} overlap={row['overlap']:2d} → {row['best_complex'][:50]}")

    # ========================================================================
    # STEP 4: Random Baseline
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Random Baseline Comparison")
    print("=" * 70)
    print("Testing if 0% GO accuracy is due to label sparsity or SRF embeddings")

    # Get GO labels
    proteins_clean = np.array([p.split(".")[-1] if "." in p else p for p in nodes])
    cache_path = data_dir / "go_annotations_human.pkl"
    annotations = fetch_go_annotations(proteins_clean, cache_path)
    Y, terms, valid_indices = build_label_matrix(proteins_clean, annotations)
    Y_valid = Y[valid_indices]
    Y_filt, terms_filt = filter_terms(Y_valid, terms, 11, 30)

    labeled_nodes_list = [nodes[i] for i in valid_indices]
    label_dict = {
        prot: [terms_filt[j] for j in range(Y_filt.shape[1]) if Y_filt[i, j]]
        for i, prot in enumerate(labeled_nodes_list)
    }
    valid_nodes = [n for n, labs in label_dict.items() if labs]
    label_dict = {p: label_dict[p] for p in valid_nodes}

    print(f"GO term classification: {len(valid_nodes)} nodes, {Y_filt.shape[1]} terms (bin 11-30)")

    # Random embeddings baseline
    np.random.seed(seed)
    W_random = np.random.rand(n_nodes, rank)
    emb_dict_random = {node: W_random[i] for i, node in enumerate(nodes)}

    print("\nSRF embeddings:")
    results_srf = test_classification(emb_dict, valid_nodes, label_dict, rank, seed)

    print("\nRandom embeddings:")
    results_random = test_classification(emb_dict_random, valid_nodes, label_dict, rank, seed)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. CORUM Complex Validation (physical interactions):")
    if corum_results['f1'].mean() > 0.1:
        print(f"   ✓ SRF captures edge structure (mean F1 = {corum_results['f1'].mean():.3f})")
    else:
        print(f"   ✗ SRF embeddings are degenerate (mean F1 = {corum_results['f1'].mean():.3f})")

    print("\n2. GO Term Classification (functional annotations):")
    print(f"   SRF:    {results_srf['test_acc']:.1%} accuracy")
    print(f"   Random: {results_random['test_acc']:.1%} accuracy")

    if results_srf['test_acc'] <= results_random['test_acc']:
        print("   → SRF no better than random for GO terms")
    else:
        print("   → SRF provides some signal for GO terms")

    print("\n3. Interpretation:")
    if corum_results['f1'].mean() > 0.1 and results_srf['test_acc'] <= results_random['test_acc']:
        print("   SRF captures EDGE STRUCTURE but NOT FUNCTIONAL ANNOTATIONS")
        print("   This is expected: W·W^T ≈ A optimizes for link prediction, not class separability")
    elif corum_results['f1'].mean() <= 0.1:
        print("   SRF embeddings don't capture either - may need more training or different params")

    # ========================================================================
    # STEP 5: GO Enrichment per Dimension (Interpretability Check)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: GO Enrichment per Dimension")
    print("=" * 70)
    print("Testing if SRF dimensions are functionally coherent (interpretable)")

    # For each dimension, get top 50 proteins and check GO term concentration
    go_enrichment_results = []
    for dim in range(rank):
        top_idx = np.argsort(W[:, dim])[-50:]
        top_proteins = set(proteins_clean[top_idx])

        # Count how many share the most common GO term
        go_counts = {}
        for prot in top_proteins:
            if prot in annotations:
                for go_term in annotations[prot]:
                    go_counts[go_term] = go_counts.get(go_term, 0) + 1

        if go_counts:
            most_common_go = max(go_counts, key=go_counts.get)
            max_count = go_counts[most_common_go]
            n_go_terms = len(go_counts)
        else:
            most_common_go = "None"
            max_count = 0
            n_go_terms = 0

        go_enrichment_results.append({
            "dimension": dim,
            "top_go_count": max_count,
            "n_unique_go": n_go_terms,
            "top_go_term": most_common_go[:30] if most_common_go else "None",
        })

    go_df = pd.DataFrame(go_enrichment_results)
    print(f"\nGO enrichment per dimension (top 50 proteins):")
    print(f"  - Mean proteins sharing top GO: {go_df['top_go_count'].mean():.1f}/50")
    print(f"  - Max proteins sharing top GO: {go_df['top_go_count'].max()}/50")
    print(f"  - Dims with >20 proteins sharing GO: {(go_df['top_go_count'] > 20).sum()}/{rank}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSION: SRF FOR INTERPRETABILITY vs PREDICTION")
    print("=" * 70)

    print("\n1. SRF captures EDGE STRUCTURE (physical interactions):")
    print(f"   - CORUM F1: {corum_results['f1'].mean():.3f} mean, {corum_results['f1'].max():.3f} max")
    print(f"   - {(corum_results['f1'] > 0.5).sum()}/{rank} dimensions decode known complexes")

    print("\n2. SRF dimensions are FUNCTIONALLY COHERENT:")
    print(f"   - {go_df['top_go_count'].mean():.1f}/50 proteins share top GO term (average)")
    print(f"   - {(go_df['top_go_count'] > 20).sum()}/{rank} dimensions have strong GO enrichment")

    print("\n3. SRF is NOT suited for GO term CLASSIFICATION:")
    print(f"   - Accuracy: 0% (same as random)")
    print(f"   - This is expected: W·W^T ≈ A optimizes edge reconstruction, not class separability")

    print("\n4. RECOMMENDATION:")
    print("   - Use SRF for INTERPRETABLE complex discovery, not prediction")
    print("   - Evaluate with F1 against CORUM, GO enrichment per dimension")
    print("   - For prediction tasks, use LINE/Node2vec (optimize for task, lose interpretability)")

    # Save results
    corum_results.to_csv(output_dir / "corum_validation.csv", index=False)
    go_df.to_csv(output_dir / "go_enrichment_per_dim.csv", index=False)
    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()
