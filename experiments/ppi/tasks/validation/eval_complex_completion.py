"""
Standard complex completion evaluation following field conventions.
Leave-one-out validation on CORUM complexes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm

from analyses.ppi.corum import load_corum
from analyses.ppi.graph_utils import load_network, build_sparse_adjacency


def compute_similarity_matrices(adj_sparse: csr_matrix) -> dict[str, csr_matrix]:
    """Compute similarity matrices for baseline methods."""
    similarities = {}

    # Common Neighbors
    similarities["CN"] = adj_sparse @ adj_sparse

    # Adamic-Adar
    degree = np.array(adj_sparse.sum(axis=1)).flatten()
    safe_degree = np.maximum(degree, 2.0)
    inv_log_degree = 1.0 / np.log(safe_degree)
    weights_diag = diags(inv_log_degree, format="csr")
    similarities["AA"] = adj_sparse @ weights_diag @ adj_sparse

    # Resource Allocation
    safe_degree = np.maximum(degree, 1.0)
    inv_degree = 1.0 / safe_degree
    weights_diag = diags(inv_degree, format="csr")
    similarities["RA"] = adj_sparse @ weights_diag @ adj_sparse

    return similarities


def evaluate_embedding_on_complex(
    embedding: np.ndarray, complex_idx: list[int], held_out_idx: int, n_proteins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate embedding-based method on single held-out protein."""
    remaining_idx = [idx for idx in complex_idx if idx != held_out_idx]

    if len(remaining_idx) == 0:
        return np.zeros(n_proteins), np.zeros(n_proteins, dtype=int)

    centroid = embedding[remaining_idx].mean(axis=0, keepdims=True)
    similarities = cosine_similarity(embedding, centroid).flatten()

    labels = np.zeros(n_proteins, dtype=int)
    labels[held_out_idx] = 1

    mask = np.ones(n_proteins, dtype=bool)
    mask[complex_idx] = False
    mask[held_out_idx] = True

    return similarities[mask], labels[mask]


def evaluate_baseline_on_complex(
    similarity_matrix: csr_matrix,
    complex_idx: list[int],
    held_out_idx: int,
    n_proteins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate baseline method on single held-out protein."""
    remaining_idx = [idx for idx in complex_idx if idx != held_out_idx]

    if len(remaining_idx) == 0:
        return np.zeros(n_proteins), np.zeros(n_proteins, dtype=int)

    scores = np.array(similarity_matrix[:, remaining_idx].sum(axis=1)).flatten()

    labels = np.zeros(n_proteins, dtype=int)
    labels[held_out_idx] = 1

    mask = np.ones(n_proteins, dtype=bool)
    mask[complex_idx] = False
    mask[held_out_idx] = True

    return scores[mask], labels[mask]


def evaluate_all_methods(
    srf_embedding: np.ndarray,
    node2vec_embedding: np.ndarray,
    adj_sparse: csr_matrix,
    proteins: np.ndarray,
    corum_complexes: dict[str, set[str]],
    min_size: int = 3,
    max_size: int = 50,
) -> pd.DataFrame:
    """Evaluate all methods on complex completion task."""
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    n_proteins = len(proteins)

    # Filter complexes by size and coverage
    valid_complexes = {}
    for name, members in corum_complexes.items():
        present = [m for m in members if m in protein_to_idx]
        if min_size <= len(present) <= max_size:
            valid_complexes[name] = present

    print(f"Evaluating {len(valid_complexes)} complexes (size {min_size}-{max_size})")

    # Precompute similarity matrices for baselines
    baseline_similarities = compute_similarity_matrices(adj_sparse)

    results = []

    for complex_name, complex_members in tqdm(
        valid_complexes.items(), desc="Complexes"
    ):
        complex_idx = [protein_to_idx[m] for m in complex_members]

        for held_out_idx in complex_idx:
            # Embedding methods
            srf_scores, labels = evaluate_embedding_on_complex(
                srf_embedding, complex_idx, held_out_idx, n_proteins
            )

            n2v_scores, _ = evaluate_embedding_on_complex(
                node2vec_embedding, complex_idx, held_out_idx, n_proteins
            )

            # Baseline methods
            cn_scores, _ = evaluate_baseline_on_complex(
                baseline_similarities["CN"], complex_idx, held_out_idx, n_proteins
            )

            aa_scores, _ = evaluate_baseline_on_complex(
                baseline_similarities["AA"], complex_idx, held_out_idx, n_proteins
            )

            ra_scores, _ = evaluate_baseline_on_complex(
                baseline_similarities["RA"], complex_idx, held_out_idx, n_proteins
            )

            # Compute metrics
            if len(np.unique(labels)) == 2:
                results.append(
                    {
                        "complex": complex_name,
                        "complex_size": len(complex_idx),
                        "srf_auroc": roc_auc_score(labels, srf_scores),
                        "srf_aupr": average_precision_score(labels, srf_scores),
                        "node2vec_auroc": roc_auc_score(labels, n2v_scores),
                        "node2vec_aupr": average_precision_score(labels, n2v_scores),
                        "cn_auroc": roc_auc_score(labels, cn_scores),
                        "cn_aupr": average_precision_score(labels, cn_scores),
                        "aa_auroc": roc_auc_score(labels, aa_scores),
                        "aa_aupr": average_precision_score(labels, aa_scores),
                        "ra_auroc": roc_auc_score(labels, ra_scores),
                        "ra_aupr": average_precision_score(labels, ra_scores),
                    }
                )

    return pd.DataFrame(results)


def main() -> None:
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    baseline_dir = base_dir / "experiments" / "ppi" / "validation" / "baselines"
    validation_output_dir = (
        base_dir
        / "experiments"
        / "ppi"
        / "validation"
        / "outputs"
        / "complex_completion"
    )
    validation_output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    srf_embedding = np.load(output_dir / "embedding.npy")
    node2vec_embedding = np.load(baseline_dir / "node2vec_embedding.npy")

    with open(output_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])

    # Load CORUM complexes
    corum_path = data_dir / "corum_complexes.txt"
    corum_complexes = load_corum(corum_path)

    # Load STRING network
    string_path = data_dir / "STRING_human_min900_v12.csv"
    g, _ = load_network(string_path)
    adj_sparse = build_sparse_adjacency(g, sorted(g.nodes()))

    # Evaluate
    results_df = evaluate_all_methods(
        srf_embedding,
        node2vec_embedding,
        adj_sparse,
        proteins,
        corum_complexes,
        min_size=3,
        max_size=50,
    )

    # Save results
    results_df.to_csv(validation_output_dir / "results.csv", index=False)

    # Print summary
    print("\nComplex Completion Results (AUROC ± std):")
    for method in ["srf", "node2vec", "cn", "aa", "ra"]:
        auroc_mean = results_df[f"{method}_auroc"].mean()
        auroc_std = results_df[f"{method}_auroc"].std()
        aupr_mean = results_df[f"{method}_aupr"].mean()
        aupr_std = results_df[f"{method}_aupr"].std()
        print(
            f"{method.upper():10s} AUROC: {auroc_mean:.3f} ± {auroc_std:.3f}  |  AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}"
        )

    print(f"\nResults saved to: {validation_output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
