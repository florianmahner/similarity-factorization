import numpy as np
from srf.mixed.admm import ADMM
from srf.simulation import add_noise_with_snr
from tools.rsa import compute_similarity
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from srf.helpers import load_spose_embedding, best_pairwise_match
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def process_single_model(
    model,
    spose_embedding,
    rsm,
    snr,
    seed,
):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    w = model.fit_transform(rsm)
    corrs = best_pairwise_match(spose_embedding, w)
    results = []
    for i, corr in enumerate(corrs):
        results.append(
            {
                "Dimension": i,
                "Correlation": corr,
                "SNR": snr,
                "Seed": seed,
            }
        )

    return results


def run_spose_reconstruction_simulation(
    model,
    spose_embedding,
    seeds=[0, 42, 123, 456, 789],
    snr=1.0,
    similarity_measure="cosine",
):

    k = spose_embedding.shape[1]

    # Apply noise once with the specified SNR
    if snr == 1.0:
        noisy_spose = spose_embedding
    else:
        noisy_spose = add_noise_with_snr(spose_embedding, snr)

    simk = compute_similarity(noisy_spose, noisy_spose, similarity_measure)

    # Create all tasks for parallel processing with different seeds
    tasks = []
    for seed in seeds:
        tasks.append((model, spose_embedding, simk, snr, seed))

    # Run all tasks in parallel
    print(f"Running {len(tasks)} tasks in parallel with different seeds...")
    all_results = Parallel(n_jobs=-1)(
        delayed(process_single_model)(*task) for task in tqdm(tasks)
    )

    # Flatten results and convert to DataFrame
    df = pd.DataFrame([item for sublist in all_results for item in sublist])

    return df


if __name__ == "__main__":
    # Load the dataset
    max_objects = 1854
    max_dims = 66

    spose_embedding = load_spose_embedding(max_objects=max_objects, max_dims=max_dims)
    model = ADMM(
        rank=max_dims,
        max_outer=100,
        w_inner=10,
        tol=0.0,
        rho=1.0,
        init="random_sqrt",
        verbose=True,
    )

    # Run the simulation with different seeds
    df = run_spose_reconstruction_simulation(
        model,
        spose_embedding,
        seeds=np.arange(30),
        snr=1.0,  # Fixed SNR (no noise)
        similarity_measure="cosine",
    )
    path = Path("/LOCAL/fmahner/srf/results/benchmarks/spose_reconstruction.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
