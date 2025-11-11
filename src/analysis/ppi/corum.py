import numpy as np
import pandas as pd
from pathlib import Path


def map_string_ids_to_genes(
    proteins: list[str], mapping_file: Path
) -> tuple[list[str], int]:
    """Map STRING protein IDs to gene names."""
    if not mapping_file.exists():
        return proteins, 0

    df = pd.read_csv(mapping_file)
    alias_map = dict(zip(df["string_protein_id"], df["preferred_name"]))

    reverse_map = {}
    for string_id, gene in alias_map.items():
        ensp_id = string_id.split(".", 1)[1] if "." in string_id else string_id
        reverse_map[ensp_id] = gene

    mapped = [reverse_map.get(p, p) if "ENSP" in p else p for p in proteins]
    mapped_count = sum(1 for o, m in zip(proteins, mapped) if o != m)

    return mapped, mapped_count


def load_corum(filepath: Path | str) -> dict[str, set[str]]:
    """Load CORUM complexes from TSV file.

    Parameters
    ----------
    filepath : Path or str
        Path to CORUM complexes TSV file

    Returns
    -------
    dict[str, set[str]]
        Dictionary mapping complex names to sets of gene names
    """
    df = pd.read_csv(filepath, sep="\t")
    complexes = {}
    for _, row in df.iterrows():
        name = row["complex_name"]
        genes = str(row["subunits_gene_name"]).split(";")
        genes = [g.strip() for g in genes if g.strip() and g.strip() != "nan"]
        if genes:
            complexes[name] = set(genes)
    return complexes


def validate_embedding_against_corum(
    embedding: np.ndarray,
    proteins: np.ndarray | list[str],
    corum_complexes: dict[str, set[str]],
    top_n: int = 5,
) -> pd.DataFrame:
    """Validate SRF embedding dimensions against CORUM complexes.

    For each dimension, finds the best matching CORUM complex and computes
    precision, recall, and F1 score.

    Parameters
    ----------
    embedding : np.ndarray
        SRF embedding matrix (n_proteins, n_dims)
    proteins : np.ndarray or list[str]
        Protein names corresponding to rows of embedding
    corum_complexes : dict[str, set[str]]
        Dictionary mapping complex names to sets of gene names
    threshold : float
        Threshold for considering a protein "active" in a dimension
    top_n : int
        Number of top proteins to consider per dimension

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: dimension, best_complex, f1, precision, recall, overlap
    """
    proteins = np.asarray(proteins)

    n_dims = embedding.shape[1]
    results = []

    for dim_idx in range(n_dims):
        loadings = embedding[:, dim_idx]

        top_k = min(top_n, len(loadings))
        top_indices = np.argsort(loadings)[-top_k:][::-1]
        top_proteins = set(proteins[top_indices])

        best_complex = None
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_overlap = 0

        for complex_name, complex_proteins in corum_complexes.items():
            overlap = len(top_proteins & complex_proteins)
            if overlap == 0:
                continue

            precision = overlap / len(top_proteins)
            recall = overlap / len(complex_proteins)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_overlap = overlap
                best_complex = complex_name

        if best_complex is None:
            best_complex = "No match"

        results.append(
            {
                "dimension": dim_idx,
                "best_complex": best_complex,
                "f1": best_f1,
                "precision": best_precision,
                "recall": best_recall,
                "overlap": best_overlap,
            }
        )

    return pd.DataFrame(results)
