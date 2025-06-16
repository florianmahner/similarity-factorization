"""
Sparse SPoSE Reconstruction Analysis

This script compares ADMM and SPoSE models for reconstructing similarity matrices
from human triplet judgments and evaluates their performance against ground truth.
Runs multiple seeds in parallel for statistical robustness.
"""

# TODO evaluate this on a low data regime compared to VICE
# TODO replace path with this repository's data directory

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import multiprocessing as mp
from joblib import Parallel, delayed

from srf.io import load_things_image_data
from srf.mixed.admm import ADMM
from tools.rsa import correlate_rsms, reconstruct_rsm_batched


# Data paths
DATA_ROOT = Path("/LOCAL/fmahner/model-comparisons/data")
SPOSE_EMBEDDING_PATH = DATA_ROOT / "misc" / "spose_embedding_66d.txt"
WORDS48_PATH = DATA_ROOT / "misc" / "words48.csv"
GROUND_TRUTH_RSM_PATH = DATA_ROOT / "misc" / "rdm48_human.mat"
TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "trainset.txt"
VALIDATION_TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "validationset.txt"

# Dataset paths
THINGS_DATASET_PATH = "/SSD/datasets/things"

# Output paths
RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "sparse_spose_recon.csv"
SUMMARY_CSV = RESULTS_DIR / "sparse_spose_recon_summary.csv"

# Model parameters
ADMM_PARAMS = {
    "rank": 66,
    "max_outer": 30,
    "w_inner": 10,
    "verbose": False,
    "tol": 0.0,
    "rho": 1.0,
    "init": "random_sqrt",
}

# Experiment parameters
N_SEEDS = 10
N_WORKERS = min(mp.cpu_count(), 10)  # 10 is roughly 512 gb mem

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
    similarity: np.ndarray, mask: np.ndarray, params: dict, seed: int = None
) -> np.ndarray:
    """Fit ADMM model to similarity data with optional seed."""
    if seed is not None:
        np.random.seed(seed)

    params["random_state"] = seed
    model = ADMM(**params)

    return model.fit_transform(
        similarity,
        mask=mask,
        bounds=(similarity[mask].min(), similarity[mask].max()),
    )


def softmax(w_i: np.ndarray, w_j: np.ndarray, w_k: np.ndarray) -> int:
    sim_ij = np.dot(w_i, w_j)
    sim_ik = np.dot(w_i, w_k)
    sim_jk = np.dot(w_j, w_k)

    similarities = np.array([sim_ij, sim_ik, sim_jk])
    exp_sims = np.exp(similarities)
    softmax_probs = exp_sims / np.sum(exp_sims)

    return np.argmax(softmax_probs)


def evaluate_triplets(
    triplets: np.ndarray,
    admm_embedding: np.ndarray,
    spose_embedding: np.ndarray,
) -> tuple[float, float]:
    """
    Evaluate the performance of the ADMM and SPoSE models on the triplets dataset.
    """

    acc_admm = 0
    acc_spose = 0
    for i, j, k in triplets:
        ooo_admm = softmax(admm_embedding[i], admm_embedding[j], admm_embedding[k])
        ooo_spose = softmax(spose_embedding[i], spose_embedding[j], spose_embedding[k])

        # the human ooo triplets are zero based
        acc_spose += ooo_spose == 0
        acc_admm += ooo_admm == 0

    acc_admm = acc_admm / len(triplets)
    acc_spose = acc_spose / len(triplets)

    return acc_admm, acc_spose


def run_single_seed(
    seed,
    spose_embedding,
    indices_48,
    rsm_48_true,
    similarity,
    mask,
    validation_triplets,
):
    """
    Run analysis for a single seed.

    Args:
        seed: Random seed for reproducibility
        All other args: Individual data components (to avoid pickling issues)

    Returns:
        Tuple of (results_list, similarity_data_list) for this seed
    """
    try:
        # Fit ADMM model with seed
        admm_embedding = fit_admm_model(similarity, mask, ADMM_PARAMS, seed=seed)

        # Compute RSMs for both models
        rsm_48_spose = reconstruct_rsm_batched(spose_embedding[indices_48])
        rsm_admm = reconstruct_rsm_batched(admm_embedding)
        rsm_48_admm = rsm_admm[np.ix_(indices_48, indices_48)]

        # Calculate correlations with ground truth
        corr_admm = correlate_rsms(rsm_48_admm, rsm_48_true)
        corr_spose = correlate_rsms(rsm_48_spose, rsm_48_true)

        # Evaluate accuracy on validation triplets
        acc_admm, acc_spose = evaluate_triplets(
            validation_triplets, admm_embedding, spose_embedding
        )

        # Prepare similarity data for scatter plots
        triu_indices = np.triu_indices(rsm_48_true.shape[0], k=1)
        true_sim = rsm_48_true[triu_indices]
        admm_sim = rsm_48_admm[triu_indices]
        spose_sim = rsm_48_spose[triu_indices]

        # Create similarity comparison data
        similarity_data = []
        for i, (true_val, admm_val) in enumerate(zip(true_sim, admm_sim)):
            similarity_data.append(
                {
                    "true_similarity": true_val,
                    "predicted_similarity": admm_val,
                    "model": "ADMM",
                    "seed": seed,
                    "pair_idx": i,
                }
            )

        # Add SPoSE data (only for seed 0 to avoid duplicates since SPoSE is deterministic)
        if seed == 0:
            for i, (true_val, spose_val) in enumerate(zip(true_sim, spose_sim)):
                similarity_data.append(
                    {
                        "true_similarity": true_val,
                        "predicted_similarity": spose_val,
                        "model": "SPoSE",
                        "seed": seed,
                        "pair_idx": i,
                    }
                )

        # Return results for this seed
        results = [
            {
                "Model": "ADMM",
                "Correlation": corr_admm,
                "Accuracy": acc_admm,
                "seed": seed,
            },
            {
                "Model": "SPoSE",
                "Correlation": corr_spose,
                "Accuracy": acc_spose,
                "seed": seed,
            },
        ]

        return results, similarity_data
    except Exception as e:
        print(f"Error in seed {seed}: {e}")
        return None, None


def load_shared_data():
    """Load all data that will be shared across seeds."""
    print("Loading shared data...")

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

    # Load validation triplets
    validation_triplets = np.loadtxt(VALIDATION_TRIPLETS_PATH).astype(int)

    return (
        spose_embedding,
        indices_48,
        rsm_48_true,
        similarity,
        mask,
        validation_triplets,
    )


def compute_summary_statistics(results_df):
    """Compute summary statistics across seeds."""
    summary_stats = []

    for model in results_df["Model"].unique():
        model_data = results_df[results_df["Model"] == model]

        for metric in ["Correlation", "Accuracy"]:
            values = model_data[metric].values
            summary_stats.append(
                {
                    "Model": model,
                    "Metric": metric,
                    "Mean": np.mean(values),
                    "Std": np.std(values),
                    "Min": np.min(values),
                    "Max": np.max(values),
                    "Median": np.median(values),
                    "Q25": np.percentile(values, 25),
                    "Q75": np.percentile(values, 75),
                    "N_Seeds": len(values),
                }
            )

    return pd.DataFrame(summary_stats)


def main():
    """Main analysis pipeline with parallel execution across seeds."""
    print(f"\nStarting analysis with {N_SEEDS} seeds using {N_WORKERS} workers...")

    spose_embedding, indices_48, rsm_48_true, similarity, mask, validation_triplets = (
        load_shared_data()
    )

    print(f"Running {N_SEEDS} seeds in parallel...")

    results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(run_single_seed)(
            seed,
            spose_embedding,
            indices_48,
            rsm_48_true,
            similarity,
            mask,
            validation_triplets,
        )
        for seed in range(N_SEEDS)
    )

    all_results = []
    all_similarity_data = []
    successful_seeds = 0
    for seed_results in results:
        if seed_results is not None and seed_results[0] is not None:
            all_results.extend(seed_results[0])
            all_similarity_data.extend(seed_results[1])
            successful_seeds += 1

    print(f"Successfully completed {successful_seeds}/{N_SEEDS} seeds")

    if not all_results:
        print("ERROR: No successful results! All seeds failed.")
        return

    results_df = pd.DataFrame(all_results)
    similarity_df = pd.DataFrame(all_similarity_data)

    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results DataFrame columns: {results_df.columns.tolist()}")
    print(f"Similarity DataFrame shape: {similarity_df.shape}")

    summary_df = compute_summary_statistics(results_df)

    RESULTS_DIR.mkdir(exist_ok=True)

    results_df.to_csv(OUTPUT_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    similarity_comparison_csv = RESULTS_DIR / "similarity_comparison.csv"
    similarity_df.to_csv(similarity_comparison_csv, index=False)

    print("\nSummary Results (Mean ± Std):")
    print("=" * 50)
    for model in summary_df["Model"].unique():
        model_stats = summary_df[summary_df["Model"] == model]
        corr_stats = model_stats[model_stats["Metric"] == "Correlation"].iloc[0]
        acc_stats = model_stats[model_stats["Metric"] == "Accuracy"].iloc[0]

        print(f"{model}:")
        print(f"  Correlation: {corr_stats['Mean']:.4f} ± {corr_stats['Std']:.4f}")
        print(f"  Accuracy:    {acc_stats['Mean']:.4f} ± {acc_stats['Std']:.4f}")

    print(f"\nDetailed results saved to: {OUTPUT_CSV}")
    print(f"Summary statistics saved to: {SUMMARY_CSV}")
    print(f"Similarity data for scatter plots saved to: {similarity_comparison_csv}")
    print(f"\nDataFrame structure for plotting:")
    print("- Detailed results have columns: Model, Correlation, Accuracy, seed")
    print("- Summary stats have columns: Model, Metric, Mean, Std, Min, Max, etc.")
    print(
        "- Similarity data has columns: true_similarity, predicted_similarity, model, seed, pair_idx"
    )


if __name__ == "__main__":
    main()
