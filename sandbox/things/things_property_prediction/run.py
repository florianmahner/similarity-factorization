"""
THINGS Property Prediction: Can SRF dimensions predict object properties?

Uses Hebart et al. THINGS dataset with 1854 objects:
- Behavioral similarity from 1.3M triplet judgments
- Property ratings (manmade, lives, natural, moves, heavy, grasp, pleasant)
- Compares SRF vs SPoSE vs ViCE for property prediction

Key question: Do SRF dimensions capture semantically meaningful properties?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pysrf import SRF, cross_val_score
from src.utils.helpers import compute_similarity_matrix_from_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROPERTY_COLS = [
    "manmade_mean", "precious_mean", "lives_mean", "heavy_mean",
    "natural_mean", "moves_mean", "grasp_mean", "hold_mean",
    "be.moved_mean", "pleasant_mean"
]

PROPERTY_NAMES = [
    "Manmade", "Precious", "Lives", "Heavy",
    "Natural", "Moves", "Graspable", "Holdable",
    "Moveable", "Pleasant"
]


def load_things_data(data_dir: Path) -> dict:
    """Load THINGS triplets, embeddings, and property ratings."""
    # Load triplets
    triplets_dir = data_dir / "triplets_147"
    train_triplets = np.loadtxt(triplets_dir / "trainset.txt", dtype=int)
    val_triplets = np.loadtxt(triplets_dir / "validationset.txt", dtype=int)

    # Load embeddings
    spose = np.loadtxt(data_dir / "spose_embedding_66d.txt")
    vice = np.loadtxt(data_dir / "vice_embedding_66d.txt")

    # Load property ratings
    props = pd.read_csv(data_dir / "things_property_ratings.csv")

    # Load concept info for alignment
    concepts = pd.read_csv(data_dir / "things_concepts.tsv", sep="\t")

    n_items = len(spose)
    log.info(f"Loaded {n_items} items")
    log.info(f"Train triplets: {len(train_triplets)}")
    log.info(f"SPoSE shape: {spose.shape}, ViCE shape: {vice.shape}")
    log.info(f"Properties: {len(props)} items x {len(PROPERTY_COLS)} properties")

    return {
        "train_triplets": train_triplets,
        "val_triplets": val_triplets,
        "spose": spose,
        "vice": vice,
        "properties": props,
        "concepts": concepts,
        "n_items": n_items,
    }


def run_srf_analysis(similarity: np.ndarray, ranks: list[int], n_repeats: int = 3) -> dict:
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

    return {
        "cv": cv,
        "best_rank": best_rank,
        "embedding": embedding,
        "model": model,
    }


def predict_properties_from_embedding(
    embedding: np.ndarray,
    properties: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    """Predict property ratings from embedding using Ridge regression with CV."""
    scaler = StandardScaler()
    X = scaler.fit_transform(embedding)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for prop_idx, prop_name in enumerate(PROPERTY_NAMES):
        y = properties[:, prop_idx]

        # Skip if too many NaNs
        valid = ~np.isnan(y)
        if valid.sum() < 100:
            continue

        X_valid = X[valid]
        y_valid = y[valid]

        # CV prediction
        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X_valid, y_valid, cv=kf)

        # Correlation between predicted and actual
        corr, pval = spearmanr(y_valid, y_pred)

        # Also fit on full data to get R²
        model.fit(X_valid, y_valid)
        r2 = model.score(X_valid, y_valid)

        results.append({
            "property": prop_name,
            "correlation": corr,
            "p_value": pval,
            "r2": r2,
            "n_valid": int(valid.sum()),
        })

    return pd.DataFrame(results)


def compare_embeddings_for_property_prediction(
    srf_emb: np.ndarray,
    spose_emb: np.ndarray,
    vice_emb: np.ndarray,
    properties: np.ndarray,
) -> pd.DataFrame:
    """Compare SRF, SPoSE, ViCE for property prediction."""
    all_results = []

    for method, emb in [("SRF", srf_emb), ("SPoSE", spose_emb), ("ViCE", vice_emb)]:
        log.info(f"Evaluating {method} (dims={emb.shape[1]})...")
        results = predict_properties_from_embedding(emb, properties)
        results["method"] = method
        results["n_dims"] = emb.shape[1]
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def find_best_predictors(
    embedding: np.ndarray,
    properties: np.ndarray,
    n_top: int = 5,
) -> pd.DataFrame:
    """For each property, find the embedding dimensions that best predict it."""
    results = []

    for prop_idx, prop_name in enumerate(PROPERTY_NAMES):
        y = properties[:, prop_idx]
        valid = ~np.isnan(y)
        if valid.sum() < 100:
            continue

        y_valid = y[valid]

        # Correlate each dimension with property
        for dim in range(embedding.shape[1]):
            x = embedding[valid, dim]
            corr, pval = spearmanr(x, y_valid)
            results.append({
                "property": prop_name,
                "dimension": dim,
                "correlation": corr,
                "abs_corr": abs(corr),
                "p_value": pval,
            })

    df = pd.DataFrame(results)

    # Get top predictors per property
    top_predictors = (
        df.sort_values("abs_corr", ascending=False)
        .groupby("property")
        .head(n_top)
    )

    return top_predictors


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "things"
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("THINGS Property Prediction Analysis")
    log.info("=" * 60)

    # Load data
    data = load_things_data(data_dir)

    # Compute similarity matrix from triplets
    log.info("\nComputing similarity matrix from triplets...")
    similarity = compute_similarity_matrix_from_triplets(
        data["n_items"],
        data["train_triplets"],
        alpha=1.0,  # Laplace smoothing
    )
    similarity = np.ascontiguousarray(similarity)

    # Handle NaN (unobserved pairs) - fill with mean similarity
    nan_mask = np.isnan(similarity)
    n_nan = nan_mask.sum()
    log.info(f"Unobserved pairs: {n_nan} ({100*n_nan/similarity.size:.1f}%)")
    similarity[nan_mask] = np.nanmean(similarity)

    # Run SRF
    log.info("\nRunning SRF analysis...")
    ranks = [10, 20, 30, 40, 50, 66]
    srf_results = run_srf_analysis(similarity, ranks=ranks, n_repeats=3)

    # Save SRF embedding
    srf_emb = srf_results["embedding"]
    np.save(output_dir / "srf_embedding.npy", srf_emb)
    log.info(f"SRF embedding shape: {srf_emb.shape}")

    # Prepare property matrix
    props_df = data["properties"]
    property_matrix = props_df[PROPERTY_COLS].values
    log.info(f"Property matrix shape: {property_matrix.shape}")

    # Compare methods
    log.info("\n" + "=" * 60)
    log.info("Property Prediction Comparison")
    log.info("=" * 60)

    comparison_df = compare_embeddings_for_property_prediction(
        srf_emb, data["spose"], data["vice"], property_matrix
    )
    comparison_df.to_csv(output_dir / "property_prediction_comparison.csv", index=False)

    # Summary by method
    log.info("\nMean correlation by method:")
    summary = comparison_df.groupby("method").agg({
        "correlation": ["mean", "std"],
        "r2": "mean",
    }).round(3)
    log.info(summary.to_string())

    # Per-property results
    log.info("\nPer-property correlations:")
    pivot = comparison_df.pivot(index="property", columns="method", values="correlation")
    log.info(pivot.round(3).to_string())
    pivot.to_csv(output_dir / "property_correlations_pivot.csv")

    # Find best predictors for SRF
    log.info("\n" + "=" * 60)
    log.info("SRF Dimension Interpretability")
    log.info("=" * 60)

    top_predictors = find_best_predictors(srf_emb, property_matrix)
    top_predictors.to_csv(output_dir / "srf_dimension_property_correlations.csv", index=False)

    # Show which dimensions predict which properties
    log.info("\nTop SRF dimensions per property:")
    for prop in PROPERTY_NAMES:
        prop_data = top_predictors[top_predictors["property"] == prop].head(3)
        if len(prop_data) > 0:
            dims = prop_data["dimension"].values
            corrs = prop_data["correlation"].values
            dim_str = ", ".join([f"D{d}({c:.2f})" for d, c in zip(dims, corrs)])
            log.info(f"  {prop}: {dim_str}")

    # Check if same dimensions predict related properties (e.g., lives & natural)
    log.info("\nDimension-property correlation matrix saved")

    # Reconstruction error
    reconstruction = srf_emb @ srf_emb.T
    recon_error = np.sqrt(np.mean((similarity - reconstruction) ** 2))
    log.info(f"\nReconstruction RMSE: {recon_error:.4f}")

    # Summary
    best_corr = comparison_df[comparison_df["method"] == "SRF"]["correlation"].max()
    mean_corr_srf = comparison_df[comparison_df["method"] == "SRF"]["correlation"].mean()
    mean_corr_spose = comparison_df[comparison_df["method"] == "SPoSE"]["correlation"].mean()

    summary_stats = {
        "n_items": data["n_items"],
        "n_triplets": len(data["train_triplets"]),
        "srf_rank": srf_results["best_rank"],
        "srf_recon_rmse": recon_error,
        "srf_mean_property_corr": mean_corr_srf,
        "spose_mean_property_corr": mean_corr_spose,
        "best_property_corr": best_corr,
    }
    pd.DataFrame([summary_stats]).to_csv(output_dir / "summary.csv", index=False)

    log.info(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
