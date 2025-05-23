import numpy as np
from srf.models.trifactor import TriFactor
from srf.simulation import add_noise_with_snr
from srf.helpers import compute_similarity
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
    snrs=[0.0, 0.25, 0.5, 1.0],
    similarity_measure="cosine",
):

    k = spose_embedding.shape[1]

    noise_data = {}
    for snr in snrs:
        if snr == 1.0:
            noisy_spose = spose_embedding
        else:
            noisy_spose = add_noise_with_snr(spose_embedding, snr)

        simk = compute_similarity(noisy_spose, noisy_spose, similarity_measure)
        if similarity_measure == "linear":
            simk = simk / simk.max()
        noise_data[snr] = (simk, spose_embedding)

    # Create all tasks for parallel processing
    tasks = []
    for i, snr in enumerate(snrs):
        simk, original_spose = noise_data[snr]
        seed = i * 100  # Unique seed for each task
        tasks.append((model, original_spose, simk, snr, seed))

    # Run all tasks in parallel
    print(f"Running {len(tasks)} tasks in parallel...")
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
    model = TriFactor(
        rank=max_dims,
        alpha=1.0,
        max_iter=300,
        verbose=True,
    )

    # Run the simulation
    df = run_spose_reconstruction_simulation(
        model, spose_embedding, snrs=[1.0], similarity_measure="linear"
    )
    path = Path("/LOCAL/fmahner/srf/results/benchmarks/spose_reconstruction.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
