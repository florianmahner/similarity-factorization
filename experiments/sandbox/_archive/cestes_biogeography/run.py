"""
Dune Meadow Biogeography: Species Co-occurrence Analysis with SRF

Analyzes species co-occurrence patterns from ecological presence/absence data.
Validates discovered factors against environmental gradients (moisture, management).

Data: Dutch dune meadow vegetation (Jongman et al. 1987, vegan package)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from scipy.stats import spearmanr, kruskal

from pysrf import SRF, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_jaccard_similarity(binary_matrix: np.ndarray) -> np.ndarray:
    """Compute Jaccard similarity between columns (species) of binary matrix."""
    n = binary_matrix.shape[1]
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            intersection = np.sum(binary_matrix[:, i] & binary_matrix[:, j])
            union = np.sum(binary_matrix[:, i] | binary_matrix[:, j])
            similarity[i, j] = intersection / union if union > 0 else 0.0

    np.fill_diagonal(similarity, 1.0)
    return similarity


def run_srf_analysis(similarity: np.ndarray, ranks: list[int], n_repeats: int = 5) -> dict:
    """Run SRF with cross-validation for rank selection."""
    log.info(f"Running cross-validation for ranks: {ranks}")

    cv = cross_val_score(
        similarity,
        param_grid={"rank": ranks},
        n_repeats=n_repeats,
        random_state=42,
        verbose=1,
    )

    best_rank = cv.best_params_["rank"]
    log.info(f"Best rank by CV: {best_rank}")

    model = SRF(
        rank=best_rank,
        rho=1.0,
        max_outer=300,
        max_inner=50,
        tol=1e-5,
        verbose=0,
        random_state=42,
    )
    embedding = model.fit_transform(similarity)
    reconstruction = embedding @ embedding.T

    return {
        "cv": cv,
        "best_rank": best_rank,
        "embedding": embedding,
        "reconstruction": reconstruction,
        "model": model,
    }


def compute_site_scores(presence_absence: np.ndarray, species_embedding: np.ndarray) -> np.ndarray:
    """Compute site scores as weighted average of species embeddings.

    Each site gets a score on each factor based on which species are present.
    """
    # Normalize presence to sum to 1 per site
    site_totals = presence_absence.sum(axis=1, keepdims=True)
    site_totals[site_totals == 0] = 1
    weights = presence_absence / site_totals

    # Site scores = weighted average of species embeddings
    site_scores = weights @ species_embedding
    return site_scores


def compare_methods(
    similarity: np.ndarray,
    presence_absence: np.ndarray,
    env_df: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame:
    """Compare SRF, NMF, PCA, and K-means on moisture prediction."""
    results = []
    moisture = env_df["Moisture"].values.astype(float)

    # 1. SRF on similarity matrix
    srf = SRF(rank=n_components, rho=1.0, max_outer=300, random_state=42, verbose=0)
    srf_emb = srf.fit_transform(similarity)
    srf_recon = srf_emb @ srf_emb.T
    srf_rmse = np.sqrt(np.mean((similarity - srf_recon) ** 2))

    # Site scores
    site_totals = presence_absence.sum(axis=1, keepdims=True)
    site_totals[site_totals == 0] = 1
    weights = presence_absence / site_totals
    srf_sites = weights @ srf_emb

    # Best moisture correlation
    srf_corrs = [abs(spearmanr(srf_sites[:, k], moisture)[0]) for k in range(n_components)]
    results.append({
        "method": "SRF",
        "recon_rmse": srf_rmse,
        "max_moisture_r": max(srf_corrs),
        "mean_moisture_r": np.mean(srf_corrs),
    })

    # 2. NMF on similarity matrix
    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    nmf_emb = nmf.fit_transform(np.clip(similarity, 0, None))
    nmf_recon = nmf_emb @ nmf.components_
    nmf_rmse = np.sqrt(np.mean((similarity - nmf_recon) ** 2))
    nmf_sites = weights @ nmf_emb
    nmf_corrs = [abs(spearmanr(nmf_sites[:, k], moisture)[0]) for k in range(n_components)]
    results.append({
        "method": "NMF",
        "recon_rmse": nmf_rmse,
        "max_moisture_r": max(nmf_corrs),
        "mean_moisture_r": np.mean(nmf_corrs),
    })

    # 3. PCA on presence/absence (transposed to get species loadings)
    pca = PCA(n_components=n_components, random_state=42)
    pca_emb = pca.fit_transform(presence_absence.T)  # species embedding
    pca_recon = pca_emb @ pca.components_ + pca.mean_
    # Can't directly compare recon since different space
    pca_sites = weights @ pca_emb
    pca_corrs = [abs(spearmanr(pca_sites[:, k], moisture)[0]) for k in range(n_components)]
    results.append({
        "method": "PCA",
        "recon_rmse": np.nan,  # Different space
        "max_moisture_r": max(pca_corrs),
        "mean_moisture_r": np.mean(pca_corrs),
    })

    # 4. K-means clustering on similarity matrix (one-hot encoding)
    kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    labels = kmeans.fit_predict(similarity)
    km_emb = np.eye(n_components)[labels]  # One-hot
    km_sites = weights @ km_emb
    km_corrs = [abs(spearmanr(km_sites[:, k], moisture)[0]) for k in range(n_components)]
    results.append({
        "method": "K-means",
        "recon_rmse": np.nan,
        "max_moisture_r": max(km_corrs),
        "mean_moisture_r": np.mean(km_corrs),
    })

    return pd.DataFrame(results)


def validate_against_environment(
    site_scores: np.ndarray,
    env_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate site factor scores against environmental variables."""
    results = []
    n_factors = site_scores.shape[1]

    # Continuous: Moisture, A1, Manure
    for var in ["Moisture", "A1", "Manure"]:
        if var in env_df.columns:
            values = env_df[var].values.astype(float)
            for k in range(n_factors):
                corr, pval = spearmanr(site_scores[:, k], values)
                results.append({
                    "variable": var,
                    "factor": k,
                    "correlation": corr,
                    "p_value": pval,
                    "test": "spearman",
                })

    # Categorical: Management, Use
    for var in ["Management", "Use"]:
        if var in env_df.columns:
            groups = env_df[var].values
            unique_groups = np.unique(groups)
            for k in range(n_factors):
                # Kruskal-Wallis test
                group_data = [site_scores[groups == g, k] for g in unique_groups]
                if all(len(g) > 0 for g in group_data):
                    stat, pval = kruskal(*group_data)
                    # Effect size: eta-squared approximation
                    n = len(site_scores)
                    eta_sq = (stat - len(unique_groups) + 1) / (n - len(unique_groups))
                    results.append({
                        "variable": var,
                        "factor": k,
                        "correlation": eta_sq,  # Using eta_sq as effect size
                        "p_value": pval,
                        "test": "kruskal",
                    })

    return pd.DataFrame(results)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "cestes"
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Dune Meadow: Species Co-occurrence Analysis")
    log.info("=" * 60)

    # Load presence/absence data
    pa_df = pd.read_csv(data_dir / "dune_meadow_pa.csv", index_col=0)
    species_names = pa_df.columns.tolist()
    presence_absence = pa_df.values.astype(int)

    # Load environmental data
    env_df = pd.read_csv(data_dir / "dune_env.csv", sep="\t", index_col=0)

    log.info(f"Dataset: {presence_absence.shape[0]} sites x {presence_absence.shape[1]} species")
    log.info(f"Environmental variables: {list(env_df.columns)}")

    # Compute species similarity from co-occurrence
    log.info("\nComputing species co-occurrence similarity (Jaccard)...")
    similarity = compute_jaccard_similarity(presence_absence)
    similarity = np.ascontiguousarray(similarity)

    # Save similarity matrix
    sim_df = pd.DataFrame(similarity, index=species_names, columns=species_names)
    sim_df.to_csv(output_dir / "species_similarity.csv")

    # Run SRF analysis
    ranks = [2, 3, 4, 5, 6]
    results = run_srf_analysis(similarity, ranks=ranks)

    # Save species embedding
    emb_df = pd.DataFrame(results["embedding"], index=species_names)
    emb_df.to_csv(output_dir / "species_embedding.csv")

    # Compute site scores from species embedding
    site_scores = compute_site_scores(presence_absence, results["embedding"])
    site_scores_df = pd.DataFrame(site_scores, index=pa_df.index)
    site_scores_df.to_csv(output_dir / "site_scores.csv")

    # Validate against environment
    log.info("\nValidating factors against environment...")
    validation_df = validate_against_environment(site_scores, env_df)
    validation_df.to_csv(output_dir / "validation.csv", index=False)

    # Top species per factor
    factor_results = []
    for k in range(results["best_rank"]):
        weights = results["embedding"][:, k]
        top_idx = np.argsort(weights)[-5:][::-1]
        for rank, idx in enumerate(top_idx):
            factor_results.append({
                "factor": k,
                "rank": rank + 1,
                "species": species_names[idx],
                "weight": weights[idx],
            })
    factor_df = pd.DataFrame(factor_results)
    factor_df.to_csv(output_dir / "factor_species.csv", index=False)

    # Reconstruction error
    recon_error = np.sqrt(np.mean((similarity - results["reconstruction"]) ** 2))
    log.info(f"Reconstruction RMSE: {recon_error:.4f}")

    # Compare methods
    log.info("\nComparing methods...")
    comparison_df = compare_methods(similarity, presence_absence, env_df, results["best_rank"])
    comparison_df.to_csv(output_dir / "method_comparison.csv", index=False)

    log.info("\nMethod Comparison:")
    log.info(f"{'Method':<10} {'RMSE':<10} {'Max |r|':<10} {'Mean |r|':<10}")
    for _, row in comparison_df.iterrows():
        rmse_str = f"{row['recon_rmse']:.4f}" if not np.isnan(row['recon_rmse']) else "N/A"
        log.info(f"{row['method']:<10} {rmse_str:<10} {row['max_moisture_r']:.3f}      {row['mean_moisture_r']:.3f}")

    # Print validation results
    log.info("\n" + "=" * 60)
    log.info("Environmental Validation:")

    # Moisture correlations (key result)
    moisture_corrs = validation_df[validation_df["variable"] == "Moisture"]
    log.info("\nMoisture gradient correlations:")
    for _, row in moisture_corrs.iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        log.info(f"  Factor {row['factor']}: r={row['correlation']:.3f} (p={row['p_value']:.4f}){sig}")

    # Management effects
    mgmt = validation_df[validation_df["variable"] == "Management"]
    log.info("\nManagement effects (Kruskal-Wallis):")
    for _, row in mgmt.iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        log.info(f"  Factor {row['factor']}: η²={row['correlation']:.3f} (p={row['p_value']:.4f}){sig}")

    # Factor interpretation
    log.info("\n" + "=" * 60)
    log.info("Factor Interpretation:")
    for k in range(results["best_rank"]):
        top = factor_df[factor_df["factor"] == k].head(3)
        species_str = ", ".join(top["species"].tolist())

        # Get moisture correlation for this factor
        moist_row = moisture_corrs[moisture_corrs["factor"] == k].iloc[0]
        moist_dir = "wet" if moist_row["correlation"] > 0 else "dry"

        log.info(f"  Factor {k} ({moist_dir}): {species_str}")

    # Summary
    best_moisture = moisture_corrs.loc[moisture_corrs["correlation"].abs().idxmax()]
    summary = {
        "n_species": len(species_names),
        "n_sites": presence_absence.shape[0],
        "best_rank": results["best_rank"],
        "recon_rmse": recon_error,
        "best_moisture_factor": int(best_moisture["factor"]),
        "best_moisture_r": best_moisture["correlation"],
        "best_moisture_p": best_moisture["p_value"],
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    log.info(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
