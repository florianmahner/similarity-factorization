"""
Complex Prediction Analysis: What Can SRF Do That LINE Cannot?

This analysis compares SRF and LINE on protein complex prediction,
not node classification. Key analyses:
1. CORUM recovery for both methods
2. Leave-one-out ablation study
3. Novel complex detection via GO enrichment
"""

from __future__ import annotations

import logging
import random
import sys
import warnings
from pathlib import Path

import hydra
import networkx as nx
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "third_party/OpenNE/src"))

from experiments.ppi.lib.utils import load_network
from experiments.ppi.lib.corum import (
    load_corum,
    map_string_ids_to_genes,
    validate_embedding_against_corum,
)
from experiments.ppi.tasks.node_classification import fit_pysrf, fetch_go_annotations

log = logging.getLogger(__name__)


def get_subgraph(g: nx.Graph, n_nodes: int) -> nx.Graph:
    if n_nodes >= len(g):
        return g
    top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:n_nodes]
    top_nodes = [n for n, _ in top_nodes]
    sub = g.subgraph(top_nodes).copy()
    largest_cc = max(nx.connected_components(sub), key=len)
    return sub.subgraph(largest_cc).copy()


def fit_line(g_nx: nx.Graph, rank: int, seed: int) -> dict[str, np.ndarray]:
    try:
        from openne.graph import Graph
        from openne.line import LINE
    except ImportError:
        log.error("OpenNE not available")
        return {}

    random.seed(seed)
    np.random.seed(seed)

    g_directed = g_nx.to_directed()
    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    g = Graph()
    g.read_g(g_directed)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    model = LINE(g, rep_size=rank, epoch=20, batch_size=500, order=3, negative_ratio=5)
    return model.vectors


def cluster_embeddings(embeddings: dict, n_clusters: int) -> dict[str, int]:
    nodes = list(embeddings.keys())
    X = np.array([embeddings[n] for n in nodes])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return dict(zip(nodes, labels))


def validate_clusters_against_corum(
    cluster_assignments: dict[str, int],
    proteins: list[str],
    corum_complexes: dict[str, set],
    n_clusters: int,
) -> pd.DataFrame:
    results = []
    for cluster_id in range(n_clusters):
        cluster_proteins = {p for p, c in cluster_assignments.items() if c == cluster_id}
        cluster_proteins = cluster_proteins & set(proteins)

        if not cluster_proteins:
            results.append({
                "dimension": cluster_id,
                "best_complex": "No match",
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "overlap": 0,
            })
            continue

        best_complex = None
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_overlap = 0

        for complex_name, complex_proteins in corum_complexes.items():
            overlap = len(cluster_proteins & complex_proteins)
            if overlap == 0:
                continue

            precision = overlap / len(cluster_proteins)
            recall = overlap / len(complex_proteins)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_overlap = overlap
                best_complex = complex_name

        results.append({
            "dimension": cluster_id,
            "best_complex": best_complex or "No match",
            "f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "overlap": best_overlap,
        })

    return pd.DataFrame(results)


def leave_one_out_evaluation(
    g_sub: nx.Graph,
    nodes: list[str],
    proteins_gene: list[str],
    corum_complexes: dict[str, set],
    rank: int,
    seed: int,
    min_complex_size: int = 5,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Leave-one-out evaluation for complex member prediction.
    For each complex with enough members in the network:
    - Remove one protein's edges
    - Fit SRF on remaining network
    - Check if removed protein ranks high in the correct dimension
    """
    protein_to_gene = dict(zip(nodes, proteins_gene))
    gene_to_protein = {g: p for p, g in protein_to_gene.items()}

    results = []
    complexes_tested = 0

    for complex_name, complex_genes in corum_complexes.items():
        complex_proteins_in_network = [
            gene_to_protein[g] for g in complex_genes if g in gene_to_protein
        ]

        if len(complex_proteins_in_network) < min_complex_size:
            continue

        complexes_tested += 1
        if complexes_tested > 10:
            break

        log.info(f"Testing leave-one-out for: {complex_name} ({len(complex_proteins_in_network)} members)")

        for left_out_protein in complex_proteins_in_network[:3]:
            g_reduced = g_sub.copy()
            edges_to_remove = list(g_reduced.edges(left_out_protein))
            g_reduced.remove_edges_from(edges_to_remove)

            nodes_reduced = sorted(g_reduced.nodes())
            edges_reduced = list(g_reduced.edges())

            try:
                embeddings = fit_pysrf(nodes_reduced, edges_reduced, rank, seed, max_outer=100)
            except Exception as e:
                log.warning(f"SRF failed for leave-one-out: {e}")
                continue

            if left_out_protein not in embeddings:
                continue

            W = np.array([embeddings[n] for n in nodes_reduced])
            proteins_arr = np.array([protein_to_gene.get(n, n) for n in nodes_reduced])

            remaining_complex = [g for g in complex_genes if g in gene_to_protein and gene_to_protein[g] != left_out_protein]
            remaining_complex_set = set(remaining_complex)

            best_dim = -1
            best_f1 = 0
            for dim in range(rank):
                top_idx = np.argsort(W[:, dim])[-top_k:]
                top_proteins = set(proteins_arr[top_idx])
                overlap = len(top_proteins & remaining_complex_set)
                if overlap > 0:
                    precision = overlap / len(top_proteins)
                    recall = overlap / len(remaining_complex_set) if remaining_complex_set else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_dim = dim

            if best_dim >= 0:
                left_out_idx = nodes_reduced.index(left_out_protein)
                left_out_loading = W[left_out_idx, best_dim]
                dim_loadings = W[:, best_dim]
                rank_percentile = (dim_loadings < left_out_loading).sum() / len(dim_loadings)

                results.append({
                    "complex": complex_name,
                    "left_out": protein_to_gene.get(left_out_protein, left_out_protein),
                    "best_dim_f1": best_f1,
                    "left_out_loading": left_out_loading,
                    "rank_percentile": rank_percentile,
                    "recovered_top_k": rank_percentile >= (1 - top_k / len(dim_loadings)),
                })

    return pd.DataFrame(results)


def detect_novel_complexes(
    W: np.ndarray,
    proteins: list[str],
    corum_validation: pd.DataFrame,
    annotations: dict[str, list[str]],
    top_k: int = 30,
    f1_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Find SRF dimensions that don't match CORUM but have GO enrichment.
    These are candidate novel complexes.
    """
    low_corum_dims = corum_validation[corum_validation["f1"] < f1_threshold]["dimension"].tolist()
    proteins_arr = np.array(proteins)

    all_go_terms = set()
    for terms in annotations.values():
        all_go_terms.update(terms)

    n_total = len(proteins)
    results = []

    for dim in low_corum_dims:
        top_idx = np.argsort(W[:, dim])[-top_k:]
        top_proteins = set(proteins_arr[top_idx])

        go_counts = {}
        for p in top_proteins:
            if p in annotations:
                for term in annotations[p]:
                    go_counts[term] = go_counts.get(term, 0) + 1

        if not go_counts:
            continue

        best_term = max(go_counts.keys(), key=lambda t: go_counts[t])
        best_count = go_counts[best_term]

        n_term_total = sum(1 for p in proteins if p in annotations and best_term in annotations[p])
        pval = hypergeom.sf(best_count - 1, n_total, n_term_total, top_k)

        results.append({
            "dimension": dim,
            "corum_f1": corum_validation[corum_validation["dimension"] == dim]["f1"].values[0],
            "best_go_term": best_term,
            "go_overlap": best_count,
            "go_pvalue": pval,
            "n_annotated_in_top": sum(1 for p in top_proteins if p in annotations),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if len(df) > 1:
        _, pvals_fdr, _, _ = multipletests(df["go_pvalue"], method="fdr_bh")
        df["go_pvalue_fdr"] = pvals_fdr
    else:
        df["go_pvalue_fdr"] = df["go_pvalue"]

    df["is_novel_candidate"] = (df["go_pvalue_fdr"] < 0.05) & (df["go_overlap"] >= 5)

    return df


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("=== Complex Prediction Analysis ===")

    data_path = Path(cfg.data_dir) / "STRING_human_min900_v12.csv"
    g_full = load_network(data_path)
    g_sub = get_subgraph(g_full, cfg.n_nodes)

    nodes = sorted(g_sub.nodes())
    edges = list(g_sub.edges())
    log.info(f"Subgraph: {len(nodes)} nodes, {len(edges)} edges")

    mapping_file = Path(cfg.data_dir) / "protein_info_900.csv"
    proteins_gene, mapped_count = map_string_ids_to_genes(nodes, mapping_file)
    log.info(f"Mapped {mapped_count} proteins to gene names")

    corum_path = Path(cfg.data_dir) / "corum_complexes.txt"
    corum_complexes = load_corum(corum_path)
    log.info(f"Loaded {len(corum_complexes)} CORUM complexes")

    output_dir = Path.cwd()

    # --- Analysis 1: SRF CORUM Validation ---
    log.info("\n--- Fitting SRF ---")
    srf_embeddings = fit_pysrf(nodes, edges, cfg.rank, cfg.seed, max_outer=200)
    W_srf = np.array([srf_embeddings[n] for n in nodes])

    srf_validation = validate_embedding_against_corum(
        W_srf, proteins_gene, corum_complexes, top_n=cfg.top_k
    )
    srf_validation["method"] = "SRF"
    srf_validation.to_csv(output_dir / "srf_corum_validation.csv", index=False)
    log.info(f"SRF CORUM F1: mean={srf_validation['f1'].mean():.3f}, max={srf_validation['f1'].max():.3f}")

    # --- Analysis 2: LINE + Clustering CORUM Validation ---
    log.info("\n--- Fitting LINE ---")
    line_embeddings = fit_line(g_sub, cfg.rank, cfg.seed)

    if line_embeddings:
        line_clusters = cluster_embeddings(line_embeddings, cfg.rank)
        proteins_in_line = [proteins_gene[nodes.index(n)] for n in line_embeddings.keys() if n in nodes]

        line_validation = validate_clusters_against_corum(
            {proteins_gene[nodes.index(n)]: c for n, c in line_clusters.items() if n in nodes},
            proteins_in_line,
            corum_complexes,
            cfg.rank,
        )
        line_validation["method"] = "LINE+KMeans"
        line_validation.to_csv(output_dir / "line_corum_validation.csv", index=False)
        log.info(f"LINE+KMeans CORUM F1: mean={line_validation['f1'].mean():.3f}, max={line_validation['f1'].max():.3f}")

        combined = pd.concat([srf_validation, line_validation], ignore_index=True)
        combined.to_csv(output_dir / "combined_corum_validation.csv", index=False)
    else:
        log.warning("LINE failed, skipping comparison")
        combined = srf_validation

    # --- Analysis 3: Leave-One-Out Ablation ---
    log.info("\n--- Leave-One-Out Ablation ---")
    loo_results = leave_one_out_evaluation(
        g_sub, nodes, proteins_gene, corum_complexes,
        cfg.rank, cfg.seed, cfg.min_complex_size, cfg.top_k
    )
    if not loo_results.empty:
        loo_results.to_csv(output_dir / "leave_one_out_results.csv", index=False)
        recovery_rate = loo_results["recovered_top_k"].mean()
        log.info(f"Leave-one-out recovery rate: {recovery_rate:.2%}")
        log.info(f"Mean rank percentile of left-out proteins: {loo_results['rank_percentile'].mean():.2%}")
    else:
        log.warning("No complexes tested in leave-one-out")

    # --- Analysis 4: Novel Complex Detection ---
    log.info("\n--- Novel Complex Detection ---")
    proteins_clean = [p.split(".")[-1] if "." in p else p for p in nodes]
    cache_path = Path(cfg.data_dir) / "go_annotations_human.pkl"
    annotations = fetch_go_annotations(np.array(proteins_clean), cache_path)

    novel_complexes = detect_novel_complexes(
        W_srf, proteins_gene, srf_validation, annotations,
        top_k=cfg.top_k, f1_threshold=0.3
    )
    if not novel_complexes.empty:
        novel_complexes.to_csv(output_dir / "novel_complex_candidates.csv", index=False)
        n_candidates = novel_complexes["is_novel_candidate"].sum()
        log.info(f"Found {n_candidates} novel complex candidates (low CORUM F1, significant GO enrichment)")
    else:
        log.warning("No novel complex candidates found")

    # --- Summary ---
    log.info("\n=== Summary ===")
    log.info(f"SRF: {srf_validation['f1'].mean():.3f} mean F1 on CORUM")
    if line_embeddings:
        log.info(f"LINE+KMeans: {line_validation['f1'].mean():.3f} mean F1 on CORUM")
    if not loo_results.empty:
        log.info(f"Leave-one-out recovery: {recovery_rate:.2%}")
    if not novel_complexes.empty:
        log.info(f"Novel complex candidates: {n_candidates}")

    log.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
