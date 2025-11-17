"""GO/KEGG enrichment per SRF factor for interpretability-focused validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
GENE_SETS = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
]
EPS = 1e-300
REPO_ROOT = Path(__file__).resolve().parents[3]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run enrichment analysis per SRF factor to highlight coherent biological "
            "signals; complements CORUM validation metrics."
        )
    )
    default_embedding = (
        REPO_ROOT
        / "experiments"
        / "ppi"
        / "outputs"
        / "corum_validation"
        / "embedding.npy"
    )
    default_proteins = (
        REPO_ROOT
        / "experiments"
        / "ppi"
        / "outputs"
        / "corum_validation"
        / "proteins.txt"
    )
    default_output = (
        REPO_ROOT
        / "experiments"
        / "ppi"
        / "validation"
        / "outputs"
        / "factor_enrichment"
    )
    default_corum = (
        REPO_ROOT
        / "experiments"
        / "ppi"
        / "outputs"
        / "corum_validation"
        / "validation_results.csv"
    )

    parser.add_argument("--embedding-path", type=Path, default=default_embedding)
    parser.add_argument("--proteins-path", type=Path, default=default_proteins)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--corum-results-path",
        type=Path,
        default=default_corum,
        help="Optional CORUM validation CSV for joining.",
    )
    parser.add_argument(
        "--top-n", type=int, default=100, help="Top loadings per factor for enrichment."
    )
    parser.add_argument(
        "--min-terms",
        type=int,
        default=5,
        help="Minimum enriched terms to keep a factor.",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.05,
        help="Adjusted P-value cutoff for calling a term significant.",
    )

    return parser.parse_args()


def get_top_proteins_per_factor(
    embedding: np.ndarray, proteins: np.ndarray, n_top: int
) -> dict[int, list[str]]:
    top_proteins: dict[int, list[str]] = {}
    for dim_idx in range(embedding.shape[1]):
        loadings = embedding[:, dim_idx]
        top_indices = np.argsort(loadings)[-n_top:][::-1]
        top_proteins[dim_idx] = proteins[top_indices].tolist()
    return top_proteins


def run_enrichment_for_factor(
    gene_list: list[str], factor_idx: int, gene_sets: list[str]
) -> pd.DataFrame:
    try:
        enrichment = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="human",
            outdir=None,
            no_plot=True,
            cutoff=0.05,
        )
    except Exception as exc:  # pragma: no cover (network errors)
        LOGGER.warning("Factor %d enrichment failed: %s", factor_idx, exc)
        return pd.DataFrame()

    if enrichment.results is None or len(enrichment.results) == 0:
        return pd.DataFrame()

    df = enrichment.results.copy()
    df["factor"] = factor_idx
    return df


def identify_interesting_factors(
    all_enrichment: pd.DataFrame, min_terms: int, significance_threshold: float
) -> list[int]:
    factor_scores = []

    for factor_idx in sorted(all_enrichment["factor"].unique()):
        factor_enr = all_enrichment[all_enrichment["factor"] == factor_idx]
        n_terms = len(factor_enr)
        sig_terms = factor_enr[factor_enr["Adjusted P-value"] <= significance_threshold]
        sig_count = len(sig_terms)
        avg_combined_score = factor_enr["Combined Score"].mean() if n_terms > 0 else 0.0
        min_pval = factor_enr["Adjusted P-value"].min() if n_terms > 0 else 1.0
        score = (
            (sig_count + 1) * np.log1p(avg_combined_score) * (-np.log10(min_pval + EPS))
        )
        factor_scores.append(
            {
                "factor": factor_idx,
                "n_terms": n_terms,
                "n_significant_terms": sig_count,
                "score": score,
            }
        )

    scores_df = pd.DataFrame(factor_scores).sort_values("score", ascending=False)
    mask = (scores_df["n_terms"] >= min_terms) & (scores_df["n_significant_terms"] > 0)
    interesting = scores_df[mask]["factor"].tolist()

    return interesting[:10]


def plot_enrichment_for_factor(
    factor_enr: pd.DataFrame, factor_idx: int, output_dir: Path
) -> None:
    if factor_enr.empty:
        LOGGER.info("Factor %d has no enrichment results; skipping plot.", factor_idx)
        return

    top_terms = factor_enr.nsmallest(20, "Adjusted P-value")
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_terms))
    ax.barh(y_pos, -np.log10(top_terms["Adjusted P-value"]), alpha=0.8, color="#2E86AB")

    labels = [
        term if len(term) <= 60 else f"{term[:57]}..." for term in top_terms["Term"]
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.set_title(f"Factor {factor_idx}: top enriched terms")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"factor_{factor_idx}_enrichment.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_summary_heatmap(
    all_enrichment: pd.DataFrame,
    interesting_factors: list[int],
    output_dir: Path,
    max_factors: int = 15,
    max_terms: int = 40,
) -> None:
    top_factors = interesting_factors[:max_factors]
    if not top_factors:
        LOGGER.info("No interesting factors available for heatmap.")
        return

    all_terms: set[str] = set()
    for factor in top_factors:
        factor_enr = all_enrichment[all_enrichment["factor"] == factor]
        top_terms = factor_enr.nsmallest(10, "Adjusted P-value")["Term"].tolist()
        all_terms.update(top_terms)

    ordered_terms = sorted(all_terms)[:max_terms]
    if not ordered_terms:
        LOGGER.info("No enriched terms available for heatmap.")
        return

    heatmap_data = np.zeros((len(ordered_terms), len(top_factors)))
    for i, term in enumerate(ordered_terms):
        for j, factor in enumerate(top_factors):
            factor_enr = all_enrichment[all_enrichment["factor"] == factor]
            term_row = factor_enr[factor_enr["Term"] == term]
            if not term_row.empty:
                pval = term_row["Adjusted P-value"].values[0]
                heatmap_data[i, j] = -np.log10(pval + EPS)

    fig, ax = plt.subplots(figsize=(12, 14))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)
    ax.set_xticks(np.arange(len(top_factors)))
    ax.set_xticklabels([f"Factor {f}" for f in top_factors], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ordered_terms)))
    ax.set_yticklabels(
        [term if len(term) <= 50 else f"{term[:47]}..." for term in ordered_terms],
        fontsize=8,
    )
    ax.set_xlabel("Factor")
    ax.set_ylabel("Enriched Term")
    ax.set_title("Factor enrichment heatmap (top factors)")

    plt.colorbar(im, ax=ax, label="-log10(Adjusted P-value)")
    plt.tight_layout()
    plt.savefig(output_dir / "enrichment_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def summarize_factor(
    factor_enr: pd.DataFrame, significance_threshold: float
) -> dict[str, object]:
    if factor_enr.empty:
        return {
            "n_terms": 0,
            "n_significant_terms": 0,
            "top_term": "None",
            "min_adjusted_p": 1.0,
            "top_combined_score": np.nan,
            "databases": "",
        }

    sig_terms = factor_enr[factor_enr["Adjusted P-value"] <= significance_threshold]
    top_term = factor_enr.nsmallest(1, "Adjusted P-value").iloc[0]
    databases = factor_enr["Gene_set"].unique().tolist()

    return {
        "n_terms": len(factor_enr),
        "n_significant_terms": len(sig_terms),
        "top_term": top_term["Term"],
        "min_adjusted_p": top_term["Adjusted P-value"],
        "top_combined_score": top_term["Combined Score"],
        "databases": ", ".join(databases),
    }


def compute_factor_statistics(
    all_enrichment: pd.DataFrame, n_factors: int, significance_threshold: float
) -> pd.DataFrame:
    stats = []
    for factor_idx in range(n_factors):
        factor_enr = all_enrichment[all_enrichment["factor"] == factor_idx]
        if factor_enr.empty:
            stats.append(
                {
                    "factor": factor_idx,
                    "n_terms": 0,
                    "n_significant_terms": 0,
                    "min_adjusted_p": 1.0,
                    "top_term": None,
                    "top_gene_set": None,
                    "top_combined_score": np.nan,
                }
            )
            continue

        sig_terms = factor_enr[factor_enr["Adjusted P-value"] <= significance_threshold]
        top_row = factor_enr.nsmallest(1, "Adjusted P-value").iloc[0]
        stats.append(
            {
                "factor": factor_idx,
                "n_terms": len(factor_enr),
                "n_significant_terms": len(sig_terms),
                "min_adjusted_p": float(top_row["Adjusted P-value"]),
                "top_term": top_row["Term"],
                "top_gene_set": top_row["Gene_set"],
                "top_combined_score": float(top_row["Combined Score"]),
            }
        )

    stats_df = pd.DataFrame(stats)
    stats_df["neg_log10_min_p"] = -np.log10(stats_df["min_adjusted_p"].clip(lower=EPS))
    return stats_df


def plot_stat_histogram(
    values: np.ndarray, xlabel: str, title: str, output_path: Path, bins: int = 30
) -> None:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        LOGGER.info("No finite values available for %s histogram.", title)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite_values, bins=bins, color="#2E86AB", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Factors")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def merge_with_corum(
    stats_df: pd.DataFrame, corum_results_path: Path, output_dir: Path
) -> None:
    if not corum_results_path.exists():
        LOGGER.info(
            "CORUM results not found at %s; skipping merge.", corum_results_path
        )
        return

    corum_df = pd.read_csv(corum_results_path)
    merged = stats_df.merge(
        corum_df, left_on="factor", right_on="dimension", how="left"
    )
    if "f1" in merged.columns:
        merged["high_confidence_complex_match"] = merged["f1"].fillna(0) >= 0.5
    merged.to_csv(output_dir / "factor_statistics_with_corum.csv", index=False)


def main() -> None:
    configure_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading embedding from %s", args.embedding_path)
    embedding = np.load(args.embedding_path)
    LOGGER.info("Embedding shape: %s", embedding.shape)

    LOGGER.info("Loading protein identifiers from %s", args.proteins_path)
    proteins = np.array(
        [line.strip() for line in args.proteins_path.read_text().splitlines()]
    )
    LOGGER.info("Number of proteins: %d", len(proteins))

    LOGGER.info("Extracting top %d proteins per factor.", args.top_n)
    top_proteins_per_factor = get_top_proteins_per_factor(
        embedding, proteins, args.top_n
    )

    LOGGER.info("Running enrichment analysis for %d factors.", embedding.shape[1])
    enrichment_rows = []
    for factor_idx in tqdm(range(embedding.shape[1]), desc="Enrichment"):
        gene_list = top_proteins_per_factor[factor_idx]
        enr_df = run_enrichment_for_factor(gene_list, factor_idx, GENE_SETS)
        if not enr_df.empty:
            enrichment_rows.append(enr_df)

    if not enrichment_rows:
        LOGGER.warning("No enrichment results found. Exiting.")
        return

    all_enrichment_df = pd.concat(enrichment_rows, ignore_index=True)
    all_enrichment_df.to_csv(args.output_dir / "all_enrichment.csv", index=False)
    LOGGER.info("Total enriched terms: %d", len(all_enrichment_df))

    stats_df = compute_factor_statistics(
        all_enrichment_df, embedding.shape[1], args.significance_threshold
    )
    stats_df.to_csv(args.output_dir / "factor_statistics.csv", index=False)

    n_factors = len(stats_df)
    factors_with_sig = int((stats_df["n_significant_terms"] > 0).sum())
    factors_with_min_terms = int(
        (stats_df["n_significant_terms"] >= args.min_terms).sum()
    )
    LOGGER.info(
        "Factors with ≥1 significant term (adj p ≤ %.3f): %d/%d (%.1f%%)",
        args.significance_threshold,
        factors_with_sig,
        n_factors,
        100 * factors_with_sig / max(n_factors, 1),
    )
    LOGGER.info(
        "Factors with ≥%d significant terms: %d/%d (%.1f%%)",
        args.min_terms,
        factors_with_min_terms,
        n_factors,
        100 * factors_with_min_terms / max(n_factors, 1),
    )

    plot_stat_histogram(
        stats_df["neg_log10_min_p"].to_numpy(),
        xlabel="-log10(min adjusted p-value)",
        title="Distribution of per-factor significance",
        output_path=args.output_dir / "factor_significance_distribution.pdf",
    )
    plot_stat_histogram(
        stats_df["n_significant_terms"].to_numpy(),
        xlabel="# significant terms (adj p ≤ threshold)",
        title="Distribution of significant term counts",
        output_path=args.output_dir / "factor_significant_term_counts.pdf",
    )

    interesting_factors = identify_interesting_factors(
        all_enrichment_df,
        min_terms=args.min_terms,
        significance_threshold=args.significance_threshold,
    )
    LOGGER.info("Top factors by enrichment signal: %s", interesting_factors)

    for factor_idx in interesting_factors[:5]:
        factor_enr = all_enrichment_df[all_enrichment_df["factor"] == factor_idx]
        plot_enrichment_for_factor(factor_enr, factor_idx, args.output_dir)

    create_summary_heatmap(
        all_enrichment_df,
        interesting_factors,
        args.output_dir,
    )

    summary_data = []
    for factor_idx in interesting_factors[:10]:
        factor_enr = all_enrichment_df[all_enrichment_df["factor"] == factor_idx]
        summary = summarize_factor(factor_enr, args.significance_threshold)
        summary["factor"] = factor_idx
        summary_data.append(summary)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)[
            [
                "factor",
                "n_terms",
                "n_significant_terms",
                "top_term",
                "min_adjusted_p",
                "top_combined_score",
                "databases",
            ]
        ]
        summary_df.to_csv(args.output_dir / "factor_summary.csv", index=False)
        LOGGER.info("Wrote summary for %d factors.", len(summary_df))
    else:
        LOGGER.info("No factors met the summary criteria.")

    merge_with_corum(stats_df, args.corum_results_path, args.output_dir)
    LOGGER.info("Factor enrichment analysis completed. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
