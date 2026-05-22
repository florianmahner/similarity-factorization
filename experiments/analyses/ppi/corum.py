import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import hypergeom, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def map_string_ids_to_genes(proteins, mapping_file):
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


def load_corum(filepath):
    df = pd.read_csv(filepath, sep="\t")
    complexes = {}
    for _, row in df.iterrows():
        name = row["complex_name"]
        genes = str(row["subunits_gene_name"]).split(";")
        genes = [g.strip() for g in genes if g.strip() and g.strip() != "nan"]
        if genes:
            complexes[name] = set(genes)
    return complexes


def validate_embedding_against_corum(embedding, proteins, corum_complexes, top_n=50):
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


def compute_additional_metrics(
    embedding, proteins, validation_df, corum_complexes, n_top=50
):
    proteins = np.asarray(proteins)
    results = []

    for dim_idx in range(embedding.shape[1]):
        loadings = embedding[:, dim_idx]
        top_idx = np.argsort(loadings)[-n_top:]
        predicted = set(proteins[top_idx])

        complex_name = validation_df.loc[
            validation_df["dimension"] == dim_idx, "best_complex"
        ].values[0]

        if complex_name not in corum_complexes:
            results.append(
                {
                    "dimension": dim_idx,
                    "geometric_accuracy": 0.0,
                    "jaccard": 0.0,
                    "mannwhitney_stat": 0.0,
                    "mannwhitney_pval": 1.0,
                }
            )
            continue

        reference = corum_complexes[complex_name]
        overlap = len(predicted & reference)

        geom_acc = (
            overlap / np.sqrt(len(predicted) * len(reference))
            if len(predicted) > 0 and len(reference) > 0
            else 0.0
        )

        union = len(predicted | reference)
        jaccard = overlap / union if union > 0 else 0.0

        in_complex = np.array([p in reference for p in proteins])
        in_loadings = loadings[in_complex]
        out_loadings = loadings[~in_complex]

        if len(in_loadings) > 0 and len(out_loadings) > 0:
            stat, pval = mannwhitneyu(in_loadings, out_loadings, alternative="greater")
        else:
            stat, pval = 0.0, 1.0

        results.append(
            {
                "dimension": dim_idx,
                "geometric_accuracy": geom_acc,
                "jaccard": jaccard,
                "mannwhitney_stat": stat,
                "mannwhitney_pval": pval,
            }
        )

    metrics_df = pd.DataFrame(results)

    reject, pvals_fdr, _, _ = multipletests(
        metrics_df["mannwhitney_pval"], method="fdr_bh"
    )
    metrics_df["mannwhitney_pval_fdr"] = pvals_fdr
    metrics_df["significant"] = reject

    return metrics_df


def compute_enrichment_pvalues(loadings, proteins, corum_proteins, n_top_range=None):
    if n_top_range is None:
        n_top_range = np.arange(10, 101, 2)

    n_total = len(proteins)
    n_complex = len(corum_proteins)
    stats = []

    for n_top in n_top_range:
        top_idx = np.argsort(loadings)[-n_top:]
        top_proteins = set(proteins[top_idx])
        overlap = len(top_proteins & corum_proteins)

        pval = hypergeom.sf(overlap - 1, n_total, n_complex, n_top)

        stats.append(
            {
                "n_top": n_top,
                "overlap": overlap,
                "pvalue": pval,
                "log10_pvalue": -np.log10(pval + 1e-300),
            }
        )

    return pd.DataFrame(stats)


def add_complex_properties(validation_df, corum_complexes):
    complex_sizes = []
    for _, row in validation_df.iterrows():
        complex_name = row["best_complex"]
        if complex_name in corum_complexes:
            complex_sizes.append(len(corum_complexes[complex_name]))
        else:
            complex_sizes.append(0)

    validation_df = validation_df.copy()
    validation_df["complex_size"] = complex_sizes
    return validation_df


def compute_summary_statistics(validation_df):
    stats = {
        "n_dimensions": len(validation_df),
        "mean_f1": validation_df["f1"].mean(),
        "median_f1": validation_df["f1"].median(),
        "std_f1": validation_df["f1"].std(),
        "max_f1": validation_df["f1"].max(),
        "min_f1": validation_df["f1"].min(),
        "n_high_f1": (validation_df["f1"] > 0.7).sum(),
        "n_moderate_f1": (
            (validation_df["f1"] > 0.5) & (validation_df["f1"] <= 0.7)
        ).sum(),
        "n_low_f1": (validation_df["f1"] <= 0.5).sum(),
        "pct_high_f1": 100 * (validation_df["f1"] > 0.7).sum() / len(validation_df),
        "mean_precision": validation_df["precision"].mean(),
        "mean_recall": validation_df["recall"].mean(),
    }
    return stats
