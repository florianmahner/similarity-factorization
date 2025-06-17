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

from utils.io import load_things_image_data
from models.admm import ADMM
from tools.rsa import correlate_rsms, reconstruct_rsm_batched


# Data paths
MAX_DIM = 49
DATA_ROOT = Path("/LOCAL/fmahner/model-comparisons/data")
SPOSE_EMBEDDING_PATH = DATA_ROOT / "misc" / f"spose_embedding_{MAX_DIM}d.txt"
WORDS48_PATH = DATA_ROOT / "misc" / "words48.csv"
GROUND_TRUTH_RSM_PATH = DATA_ROOT / "misc" / "rdm48_human.mat"
TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "trainset.txt"
VALIDATION_TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "validationset.txt"

# Dataset paths
THINGS_DATASET_PATH = "/SSD/datasets/things"

# Output paths
RESULTS_DIR = Path(f"results/spose/{MAX_DIM}")
OUTPUT_CSV = RESULTS_DIR / f"sparse_spose_recon.csv"
SUMMARY_CSV = RESULTS_DIR / f"sparse_spose_recon_summary.csv"
LOW_DATA_CSV = RESULTS_DIR / f"sparse_spose_recon_low_data.csv"
LOW_DATA_SUMMARY_CSV = RESULTS_DIR / f"sparse_spose_recon_low_data_summary.csv"

# Model parameters
ADMM_PARAMS = {
    "rank": MAX_DIM,
    "max_outer": 20,
    "w_inner": 10,
    "verbose": False,
    "tol": 0.0,
    "rho": 1.0,
    "init": "random",
}

# Experiment parameters
N_SEEDS = 10
N_WORKERS = min(mp.cpu_count(), 10)  # 10 is roughly 512 gb mem

# Low data regime parameters
DATA_PERCENTAGES = [0.05, 0.10, 0.20, 0.50, 1.0]  # 5%, 10%, 20%, 50%, 100%

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


def sample_triplets(triplets: np.ndarray, percentage: float, seed: int) -> np.ndarray:
    """
    Sample a percentage of triplets with a given random seed.

    Args:
        triplets: Original triplets array
        percentage: Fraction of triplets to sample (0.0 to 1.0)
        seed: Random seed for reproducible sampling

    Returns:
        Sampled triplets array
    """
    if percentage >= 1.0:
        return triplets

    np.random.seed(seed)
    n_samples = int(len(triplets) * percentage)
    indices = np.random.choice(len(triplets), size=n_samples, replace=False)
    return triplets[indices]


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

    # Create a copy of params to avoid modifying the shared dictionary
    local_params = params.copy()
    local_params["random_state"] = seed

    model = ADMM(**local_params)

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


def reconstruct_admm_rsm(w):
    """
    Reconstruct ADMM embedding similarity matrix to [0,1] range. Since it works on inner products,
    we need to normalize it to [0,1] range.
    Prefer min max normalization over cosine normalization or clipping since it preserves the
    relationships between items (monotonicity).
    """
    sim = w @ w.T
    sim_copy = sim.copy()
    np.fill_diagonal(sim_copy, 0)

    min_val, max_val = sim_copy.min(), sim_copy.max()
    normalized = (
        (sim - min_val) / (max_val - min_val) if max_val > min_val else sim.copy()
    )
    np.fill_diagonal(normalized, 1)
    return normalized


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
        # NOTE we use inner products instead of softmax like reconstruciton,
        # since this is what we optimize for an SPOSE used triplet based optimization
        rsm_admm = reconstruct_admm_rsm(admm_embedding)
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


def run_low_data_experiment(
    seed,
    data_percentage,
    full_triplets,
    spose_embedding,
    validation_triplets,
):
    """
    Run low data regime experiment for a single seed and data percentage.

    Args:
        seed: Random seed for reproducibility
        data_percentage: Fraction of training data to use (0.0 to 1.0)
        full_triplets: Full training triplets array
        spose_embedding: SPoSE embedding for comparison
        validation_triplets: Validation triplets for accuracy evaluation

    Returns:
        Dictionary with results for this seed and data percentage
    """
    try:
        # Sample triplets for this data percentage and seed
        sampled_triplets = sample_triplets(full_triplets, data_percentage, seed)

        # Compute similarity matrix from sampled triplets
        similarity, mask = compute_similarity_matrix(N_ITEMS, sampled_triplets)

        # Skip if no valid pairs (can happen with very low data percentages)
        if mask.sum() == 0:
            print(
                f"Warning: No valid pairs for seed {seed}, percentage {data_percentage:.1%}"
            )
            return None

        # Fit ADMM model with seed
        admm_embedding = fit_admm_model(similarity, mask, ADMM_PARAMS, seed=seed)

        # Evaluate accuracy on validation triplets (only ADMM since SPoSE is constant)
        acc_admm = 0
        for i, j, k in validation_triplets:
            ooo_admm = softmax(admm_embedding[i], admm_embedding[j], admm_embedding[k])
            acc_admm += ooo_admm == 0
        acc_admm = acc_admm / len(validation_triplets)

        # Return results for this seed and data percentage
        return {
            "Model": "ADMM",
            "Data_Percentage": data_percentage,
            "N_Triplets": len(sampled_triplets),
            "Accuracy": acc_admm,
            "seed": seed,
        }

    except Exception as e:
        print(
            f"Error in low data experiment seed {seed}, percentage {data_percentage:.1%}: {e}"
        )
        return None


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
        triplets,  # Add full triplets for low data experiments
    )


def compute_summary_statistics(results_df):
    """Compute summary statistics across seeds."""
    summary_stats = []

    for model in results_df["Model"].unique():
        model_data = results_df[results_df["Model"] == model]

        for metric in ["Correlation", "Accuracy"]:
            if metric in model_data.columns:
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


def compute_low_data_summary_statistics(results_df):
    """Compute summary statistics for low data regime experiments."""
    summary_stats = []

    for data_percentage in results_df["Data_Percentage"].unique():
        percentage_data = results_df[results_df["Data_Percentage"] == data_percentage]

        values = percentage_data["Accuracy"].values
        summary_stats.append(
            {
                "Data_Percentage": data_percentage,
                "Mean_Accuracy": np.mean(values),
                "Std_Accuracy": np.std(values),
                "Min_Accuracy": np.min(values),
                "Max_Accuracy": np.max(values),
                "Median_Accuracy": np.median(values),
                "Q25_Accuracy": np.percentile(values, 25),
                "Q75_Accuracy": np.percentile(values, 75),
                "Mean_N_Triplets": np.mean(percentage_data["N_Triplets"].values),
                "N_Seeds": len(values),
            }
        )

    return pd.DataFrame(summary_stats)


def main():
    """Main analysis pipeline with parallel execution across seeds."""
    print(f"\nStarting analysis with {N_SEEDS} seeds using {N_WORKERS} workers...")

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (
        spose_embedding,
        indices_48,
        rsm_48_true,
        similarity,
        mask,
        validation_triplets,
        full_triplets,
    ) = load_shared_data()

    print(f"Running main analysis with {N_SEEDS} seeds in parallel...")

    # Run main analysis
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

    # Run low data regime experiments
    print(f"\nRunning low data regime experiments...")
    print(f"Data percentages: {[f'{p:.1%}' for p in DATA_PERCENTAGES]}")

    low_data_jobs = []
    for seed in range(N_SEEDS):
        for data_percentage in DATA_PERCENTAGES:
            low_data_jobs.append((seed, data_percentage))

    print(f"Running {len(low_data_jobs)} low data experiments in parallel...")

    low_data_results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(run_low_data_experiment)(
            seed,
            data_percentage,
            full_triplets,
            spose_embedding,
            validation_triplets,
        )
        for seed, data_percentage in low_data_jobs
    )

    # Process low data results
    low_data_valid_results = [r for r in low_data_results if r is not None]
    print(
        f"Successfully completed {len(low_data_valid_results)}/{len(low_data_jobs)} low data experiments"
    )

    if low_data_valid_results:
        low_data_df = pd.DataFrame(low_data_valid_results)
        low_data_summary_df = compute_low_data_summary_statistics(low_data_df)
    else:
        print("Warning: No successful low data regime results!")
        low_data_df = pd.DataFrame()
        low_data_summary_df = pd.DataFrame()

    results_df.to_csv(OUTPUT_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    similarity_comparison_csv = RESULTS_DIR / "similarity_comparison.csv"
    similarity_df.to_csv(similarity_comparison_csv, index=False)

    if not low_data_df.empty:
        low_data_df.to_csv(LOW_DATA_CSV, index=False)
        low_data_summary_df.to_csv(LOW_DATA_SUMMARY_CSV, index=False)

    # Print results
    print("\nMain Analysis Results (Mean ± Std):")
    print("=" * 50)
    for model in summary_df["Model"].unique():
        model_stats = summary_df[summary_df["Model"] == model]
        corr_stats = model_stats[model_stats["Metric"] == "Correlation"].iloc[0]
        acc_stats = model_stats[model_stats["Metric"] == "Accuracy"].iloc[0]

        print(f"{model}:")
        print(f"  Correlation: {corr_stats['Mean']:.4f} ± {corr_stats['Std']:.4f}")
        print(f"  Accuracy:    {acc_stats['Mean']:.4f} ± {acc_stats['Std']:.4f}")

    if not low_data_summary_df.empty:
        print("\nLow Data Regime Results (ADMM Accuracy):")
        print("=" * 50)
        for _, row in low_data_summary_df.iterrows():
            print(
                f"{row['Data_Percentage']:.1%} data ({row['Mean_N_Triplets']:.0f} triplets): "
                f"{row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}"
            )

    print(f"\nDetailed results saved to: {OUTPUT_CSV}")
    print(f"Summary statistics saved to: {SUMMARY_CSV}")
    print(f"Similarity data for scatter plots saved to: {similarity_comparison_csv}")
    if not low_data_df.empty:
        print(f"Low data regime results saved to: {LOW_DATA_CSV}")
        print(f"Low data regime summary saved to: {LOW_DATA_SUMMARY_CSV}")
    print(f"\nDataFrame structure for plotting:")
    print("- Detailed results have columns: Model, Correlation, Accuracy, seed")
    print("- Summary stats have columns: Model, Metric, Mean, Std, Min, Max, etc.")
    print(
        "- Similarity data has columns: true_similarity, predicted_similarity, model, seed, pair_idx"
    )
    if not low_data_df.empty:
        print(
            "- Low data results have columns: Model, Data_Percentage, N_Triplets, Accuracy, seed"
        )
        print(
            "- Low data summary has columns: Data_Percentage, Mean_Accuracy, Std_Accuracy, etc."
        )


if __name__ == "__main__":
    main()
