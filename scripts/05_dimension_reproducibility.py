import numpy as np
from experiments.dimension_reliability import run_reproducibility_analysis

# Example usage
if __name__ == "__main__":
    # Example parameters
    admm_params = {
        "rank": 49,  # As mentioned in the paper
        "rho": 1.0,
        "max_outer": 100,
        "max_inner": 5,
        "tol": 0.0,
        "init": "random_sqrt",
    }
    from pathlib import Path
    from experiments.spose import compute_similarity_matrix

    MAX_DIM = 49
    DATA_ROOT = Path("/LOCAL/fmahner/model-comparisons/data")
    SPOSE_EMBEDDING_PATH = DATA_ROOT / "misc" / f"spose_embedding_{MAX_DIM}d.txt"
    WORDS48_PATH = DATA_ROOT / "misc" / "words48.csv"
    GROUND_TRUTH_RSM_PATH = DATA_ROOT / "misc" / "rdm48_human.mat"
    TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "trainset.txt"

    # Load similarity matrix and mask
    triplets = np.loadtxt(TRIPLETS_PATH).astype(int)
    similarity_matrix, mask = compute_similarity_matrix(1854, triplets)

    # Run analysis exactly as described in the paper
    results = run_reproducibility_analysis(
        similarity_matrix=similarity_matrix,
        mask=mask,
        admm_params=admm_params,
        n_runs=20,  # As in the paper
        n_jobs=-1,
        output_path="results/dimension_reproducibility.csv",
    )
