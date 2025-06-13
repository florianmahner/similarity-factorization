"""
Sparse SPoSE Reconstruction Analysis

This script compares ADMM and SPoSE models for reconstructing similarity matrices
from human triplet judgments and evaluates their performance against ground truth.
"""

# TODO evaluate this also on triplets comparing SPoSE/VICE in terms of accuracy
# TODO evaluate this on a low data regime compared to VICE

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

from srf.io import load_things_image_data
from srf.mixed.admm import ADMM
from tools.rsa import correlate_rsms, reconstruct_rsm_batched


# Data paths
DATA_ROOT = Path("/LOCAL/fmahner/model-comparisons/data")
SPOSE_EMBEDDING_PATH = DATA_ROOT / "misc" / "spose_embedding_66d.txt"
WORDS48_PATH = DATA_ROOT / "misc" / "words48.csv"
GROUND_TRUTH_RSM_PATH = DATA_ROOT / "misc" / "rdm48_human.mat"
TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "trainset.txt"

# Dataset paths
THINGS_DATASET_PATH = "/SSD/datasets/things"

# Output paths
RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "sparse_spose_recon.csv"

# Model parameters
ADMM_PARAMS = {
    "rank": 66,
    "max_outer": 30,
    "w_inner": 10,
    "verbose": True,
    "tol": 0.0,
    "rho": 1.0,
    "init": "nndsvdar",
}

# Data parameters
N_ITEMS = 1854
GROUND_TRUTH_RSM_KEY = "RDM48_triplet"

# Category name replacements
CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}


def load_embeddings(embedding_path: Path) -> np.ndarray:
    """Load and preprocess embeddings."""
    embedding = np.loadtxt(embedding_path)
    return np.maximum(embedding, 0)


def load_concept_mappings(
    words_path: Path, things_path: str, replacements: dict
) -> list:
    """Load concept names and create category mappings."""
    words48 = pd.read_csv(words_path)
    cls_names = words48["Word"].values

    # Load THINGS dataset images and extract categories
    images = load_things_image_data(things_path, filter_behavior=True)
    categories = [" ".join(Path(f).stem.split("_")[0:-1]) for f in images]

    # Apply category name replacements
    cls_names = [replacements.get(name, name) for name in cls_names]

    # Get indices for the 48 concepts
    indices_48 = [categories.index(c) for c in cls_names]

    return indices_48


def load_ground_truth(ground_truth_path: Path, rsm_key: str) -> np.ndarray:
    """Load ground truth RSM."""
    rdm_48_true = loadmat(ground_truth_path)[rsm_key]
    return 1 - rdm_48_true


def compute_similarity_matrix_v0(
    n: int, triplets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    counts = np.zeros((n, n))

    # Accumulate counts for each similar pair
    for i, j, k in triplets:
        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1

    # Create mask for observed pairs and normalize similarity matrix
    mask = (counts > 0).astype(np.uint8)
    np.fill_diagonal(mask, 0)
    similarity = counts / counts.max()

    return similarity, mask


def compute_similarity_matrix(
    n: int, triplets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Your approach with per-pair normalization."""
    counts = np.zeros((n, n))
    pair_opportunities = np.zeros(
        (n, n)
    )  # How many times each pair could have been chosen

    for i, j, k in triplets:
        # Count opportunities for each pair to be chosen as "most similar"
        if i != j:
            pair_opportunities[i, j] += 1
            pair_opportunities[j, i] += 1
        if i != k:
            pair_opportunities[i, k] += 1
            pair_opportunities[k, i] += 1
        if j != k:
            pair_opportunities[j, k] += 1
            pair_opportunities[k, j] += 1

        # Only (i,j) actually gets chosen
        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1

    # Normalize by opportunities rather than global max
    mask = (pair_opportunities > 0).astype(np.uint8)
    np.fill_diagonal(mask, 0)

    similarity = np.divide(
        counts,
        pair_opportunities,
        out=np.zeros_like(counts),
        where=pair_opportunities != 0,
    )

    return similarity, mask


def fit_admm_model(
    similarity: np.ndarray, mask: np.ndarray, params: dict
) -> np.ndarray:
    """Fit ADMM model to similarity data."""
    model = ADMM(**params)

    return model.fit_transform(
        similarity,
        mask=mask,
        bounds=(similarity[mask].min(), similarity[mask].max()),
    )


def main():
    """Main analysis pipeline."""
    print("\nLoading data...")

    # Load embeddings and mappings
    spose_embedding = load_embeddings(SPOSE_EMBEDDING_PATH)
    indices_48 = load_concept_mappings(
        WORDS48_PATH, THINGS_DATASET_PATH, CATEGORY_REPLACEMENTS
    )
    rsm_48_true = load_ground_truth(GROUND_TRUTH_RSM_PATH, GROUND_TRUTH_RSM_KEY)

    # Load triplet data and compute similarity matrix
    print("Processing triplet data...")
    triplets = np.loadtxt(TRIPLETS_PATH).astype(int)
    similarity, mask = compute_similarity_matrix(N_ITEMS, triplets)

    # Fit ADMM model
    print("Fitting ADMM model...")
    admm_embedding = fit_admm_model(similarity, mask, ADMM_PARAMS)

    # Compute RSMs for both models
    print("Computing correlations...")
    rsm_48_spose = reconstruct_rsm_batched(spose_embedding[indices_48])
    rsm_admm = reconstruct_rsm_batched(admm_embedding)
    rsm_48_admm = rsm_admm[np.ix_(indices_48, indices_48)]

    # Calculate correlations with ground truth
    corr_admm = correlate_rsms(rsm_48_admm, rsm_48_true)
    corr_spose = correlate_rsms(rsm_48_spose, rsm_48_true)

    # Create and save results
    results_df = pd.DataFrame(
        [
            {"Model": "ADMM", "Correlation": corr_admm},
            {"Model": "SPoSE", "Correlation": corr_spose},
        ]
    )

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    # Print results
    print("\nResults:")
    print(f"ADMM Correlation:  {corr_admm:.4f}")
    print(f"SPoSE Correlation: {corr_spose:.4f}")
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
