"""Variance partitioning: Factorize residuals not explained by VGG16.

1. Load VGG16 features & Human embedding (1854 concepts)
2. Ridge regression: VGG16 → Human embedding
3. Residuals = Human - Predicted
4. Build RSM from residuals (dot product)
5. SRF factorization → dimensions VGG16 can't explain
"""

import sys
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Add object-dimensions to path
sys.path.insert(0, "/LOCAL/fmahner/object-dimensions")
from objdim.utils import load_sparse_codes, load_image_data

# Add similarity-factorization to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pysrf import SRF

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

BASE_PATH = Path("/LOCAL/fmahner/object-dimensions")


def load_data():
    """Load VGG16 features and Human embedding, filtered to 1854 concepts."""
    # Get indices for 1854 behavioral concepts
    _, indices = load_image_data(
        BASE_PATH / "data/images/things", filter_behavior=True
    )
    log.info(f"Filtered to {len(indices)} concepts")

    # Load VGG16 features (24k images -> filter to 1854)
    vgg_features = np.load(
        BASE_PATH / "data/features/vgg16_bn/classifier.4/features.npy"
    )
    vgg_features = vgg_features[indices]
    log.info(f"VGG16 features: {vgg_features.shape}")

    # Load Human embedding (already 1854)
    human_embedding = load_sparse_codes(BASE_PATH / "data/embeddings/human_behavior")
    log.info(f"Human embedding: {human_embedding.shape}")

    return vgg_features, human_embedding


def predict_embedding(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Ridge regression: X → Y, return predictions, residuals, R² per dimension."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use fixed alpha (typical for high-dim) - faster than CV
    model = Ridge(alpha=1000)

    # Cross-validated predictions to avoid overfitting
    predictions = cross_val_predict(model, X_scaled, Y, cv=5)

    # Compute R² per dimension
    r2_scores = []
    for dim in range(Y.shape[1]):
        ss_res = np.sum((Y[:, dim] - predictions[:, dim]) ** 2)
        ss_tot = np.sum((Y[:, dim] - Y[:, dim].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_scores.append(r2)

    log.info(f"  R² range: [{min(r2_scores):.3f}, {max(r2_scores):.3f}]")

    residuals = Y - predictions
    return predictions, residuals, r2_scores


def main():
    log.info("=" * 60)
    log.info("Residual Factorization: What VGG16 Can't Explain")
    log.info("=" * 60)

    # Load data
    log.info("\n[1] Loading data...")
    vgg_features, human_embedding = load_data()

    # Predict human embedding from VGG16
    log.info("\n[2] Ridge regression: VGG16 → Human embedding...")
    predictions, residuals, r2_scores = predict_embedding(vgg_features, human_embedding)
    log.info(f"  Mean R²: {np.mean(r2_scores):.3f}")
    log.info(f"  Residual variance: {np.var(residuals):.4f}")

    # Build RSMs
    log.info("\n[3] Building RSMs...")

    # Original RSM (human embedding)
    rsm_original = human_embedding @ human_embedding.T
    rsm_original = rsm_original / rsm_original.max()

    # Predicted RSM
    rsm_predicted = predictions @ predictions.T
    rsm_predicted = rsm_predicted / rsm_predicted.max()

    # Residual RSM
    rsm_residual = residuals @ residuals.T
    rsm_residual = rsm_residual / rsm_residual.max()

    log.info(f"  Original RSM range: [{rsm_original.min():.3f}, {rsm_original.max():.3f}]")
    log.info(f"  Residual RSM range: [{rsm_residual.min():.3f}, {rsm_residual.max():.3f}]")

    # Factorize residual RSM
    log.info("\n[4] SRF factorization of residual RSM...")

    # Shift to non-negative for SRF
    rsm_residual_shifted = rsm_residual - rsm_residual.min()

    model = SRF(rank=10, max_outer=200, tol=1e-4, verbose=1, random_state=42)
    residual_embedding = model.fit_transform(rsm_residual_shifted)

    reconstruction = residual_embedding @ residual_embedding.T
    rmse = np.sqrt(np.mean((rsm_residual_shifted - reconstruction) ** 2))
    log.info(f"  Reconstruction RMSE: {rmse:.4f}")

    # Save results
    log.info("\n[5] Saving results...")
    np.save(OUTPUT_DIR / "human_embedding.npy", human_embedding)
    np.save(OUTPUT_DIR / "predictions.npy", predictions)
    np.save(OUTPUT_DIR / "residuals.npy", residuals)
    np.save(OUTPUT_DIR / "rsm_original.npy", rsm_original)
    np.save(OUTPUT_DIR / "rsm_residual.npy", rsm_residual)
    np.save(OUTPUT_DIR / "residual_embedding.npy", residual_embedding)
    np.save(OUTPUT_DIR / "r2_scores.npy", np.array(r2_scores))

    log.info(f"\nResults saved to: {OUTPUT_DIR}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("Summary:")
    log.info(f"  VGG16 explains {np.mean(r2_scores)*100:.1f}% of human embedding variance")
    log.info(f"  Residual embedding has {residual_embedding.shape[1]} dimensions")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
