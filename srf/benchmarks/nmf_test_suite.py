import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import itertools
from ..simulation import generate_simulation_data, add_noise_with_snr
from sklearn.decomposition import NMF
from .image_clustering import purity_score, compute_entropy, map_labels_with_hungarian
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tools.rsa import compute_similarity
from tools.rsa import correlate_rsms
from tools.stats import average_pearson_r

from mcomp.synmf.trifactor import TriFactorANLS
from mcomp.synmf.anls import SymmetricANLS, SymmetricANLSMixed

from sklearn.metrics import linear_sum_assignment
from ..utils import load_spose_embedding
from tqdm import tqdm

# TODO Write this model generic, eg we have a model that we use.


def normalize_columns(mat):
    # Normalize columns: subtract mean and divide by std (using ddof=1)
    means = np.mean(mat, axis=0, keepdims=True)
    stds = np.std(mat, axis=0, ddof=1, keepdims=True)
    return (mat - means) / stds


def compute_metrics(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    purity = purity_score(true_labels, predicted_labels)
    entropy = compute_entropy(true_labels, predicted_labels)
    return ari, nmi, purity, entropy


def greedy_match(embeddings, spose_embedding):
    """
    For each dimension (column) in 'embeddings', find the best matching dimension in 'spose_embedding'
    based on highest Pearson correlation (without replacement). Returns a list of tuples (i, j, corr)
    where i is the index in embeddings, j is the index in spose_embedding, and corr is the Pearson correlation.
    """
    # Normalize each column to obtain z-scores.
    A_norm = normalize_columns(embeddings)  # shape: (n_samples, n_dims1)
    B_norm = normalize_columns(spose_embedding)  # shape: (n_samples, n_dims2)

    n_samples = embeddings.shape[0]
    # The dot product of two z-scored columns equals (n_samples-1)*r, where r is the Pearson correlation.
    # Thus, we divide by (n_samples-1) to obtain the actual Pearson correlation coefficients.
    corr_matrix = (A_norm.T @ B_norm) / (n_samples - 1)  # shape: (n_dims1, n_dims2)

    # Keep track of the original indices for later mapping.
    a_indices = list(range(embeddings.shape[1]))
    b_indices = list(range(spose_embedding.shape[1]))

    matches = []
    # Continue matching until one set runs out of dimensions.
    while corr_matrix.shape[0] > 0 and corr_matrix.shape[1] > 0:
        # Find the index (i,j) of the maximum value in the current correlation matrix.
        i_rel, j_rel = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        best_corr = corr_matrix[i_rel, j_rel]

        # Map back to the original dimension indices.
        i_orig = a_indices[i_rel]
        j_orig = b_indices[j_rel]
        matches.append((best_corr, i_orig, j_orig))

        # Remove the matched row and column from the correlation matrix.
        corr_matrix = np.delete(corr_matrix, i_rel, axis=0)
        corr_matrix = np.delete(corr_matrix, j_rel, axis=1)

        # Update our index lists so that future matches refer to the correct original dimensions.
        del a_indices[i_rel]
        del b_indices[j_rel]

    return matches


def process_single_model(
    model,
    spose_embedding,
    rsm,
):
    w = model.fit_transform(rsm)
    matches = greedy_match(spose_embedding, w)
    corrs = [m[0] for m in matches]
    results = []
    for i, corr in enumerate(corrs):
        results.append(
            {
                "Dimension": i,
                "Correlation": corr,
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
        noisy_spose = add_noise_with_snr()(spose_embedding, snr)
        simk = compute_similarity(noisy_spose, noisy_spose, similarity_measure)
        noise_data[snr] = (simk, spose_embedding)

    # Create all tasks for parallel processing
    tasks = []
    for i, snr in enumerate(snrs):
        simk, original_spose = noise_data[snr]
        seed = i * 100  # Unique seed for each task
        tasks.append((model, simk, original_spose, snr, seed))

    # Run all tasks in parallel
    print(f"Running {len(tasks)} tasks in parallel...")
    all_results = Parallel(n_jobs=-1)(
        delayed(process_single_model)(*task) for task in tqdm(tasks)
    )

    # Flatten results and convert to DataFrame
    df = pd.DataFrame([item for sublist in all_results for item in sublist])

    return df


def run_model_comparison_simulation(
    model,
    simulation_params,
    seeds=range(10),
    noise_levels=np.linspace(0.0, 1.0, 10),
    similarity_measure="cosine",
):
    results = []

    def process_combination(noise, seed):

        simulation_params.snr = noise
        simulation_params.rng_state = seed
        X, M = generate_simulation_data(simulation_params)

        # Ground-truth labels from membership matrix M
        true_labels = np.argmax(M, axis=1)

        X_shifted = X - X.min() if X.min() < 0 else X
        S = compute_similarity(X, X, similarity_measure)
        models = {
            "NMF X": {
                "data": X_shifted,
                "model": NMF(
                    n_components=simulation_params.k,
                    init="random",
                    solver="mu",
                    random_state=seed,
                    max_iter=1000,
                    tol=0.0,
                ),
            },
            "SymNMF": {
                "data": S,
                "model": model,
            },
        }
        metrics = ["ARI", "NMI", "Accuracy", "Entropy"]
        local_results = []

        for key, model_info in models.items():
            data_shifted = (
                model_info["data"] - model_info["data"].min()
                if model_info["data"].min() < 0
                else model_info["data"]
            )

            if key == "KMeans X" or key == "Spectral Clustering":
                labels = model_info["model"].fit_predict(data_shifted)
            else:
                W = model_info["model"].fit_transform(data_shifted)
                labels = np.argmax(W, axis=1)

            labels = map_labels_with_hungarian(true_labels, labels)
            ari, nmi, purity, entropy = compute_metrics(true_labels, labels)

            for metric, value in zip(metrics, [ari, nmi, purity, entropy]):
                local_results.append(
                    {
                        "SNR": noise,
                        "Seed": seed,
                        "Model": key,
                        "Metric": metric,
                        "Value": value,
                    }
                )
        return local_results

    results = Parallel(n_jobs=-1)(
        delayed(process_combination)(noise, seed)
        for noise, seed in itertools.product(noise_levels, seeds)
    )

    # Flatten the list of results
    results = [item for sublist in results for item in sublist]

    return pd.DataFrame(results)


# ------ RSA Model tests ------#


def aligned_latent_correlation_by_dimension(x, w):
    """we find the best alignment between the latent dimensions of x and w"""
    # Ensure inputs are numpy arrays
    x, w = np.asarray(x), np.asarray(w)
    rank = x.shape[1]

    # Compute the correlation matrix between each pair of columns
    corr_matrix = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            corr_matrix[i, j] = np.corrcoef(x[:, i], w[:, j])[0, 1]

    # Use the Hungarian algorithm to maximize the sum of absolute correlations.
    # Since linear_sum_assignment minimizes cost, we use negative absolute correlations.
    cost_matrix = -np.abs(corr_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute the mean correlation for the best aligned columns
    aligned_corrs = [corr_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    mean_corr = average_pearson_r(aligned_corrs)
    return mean_corr


def run_rsa_hypothesis(model, params, similarity_measure="cosine", rsm_corr="spearman"):
    """Run simulation for a single parameter combination."""
    max_obj, latent_dim, noise, seed = params
    np.random.seed(seed)

    x = load_spose_embedding(max_obj, latent_dim)
    x_noise = add_noise(x, noise, rng=seed)
    s_noise = compute_similarity(x_noise, x_noise, similarity_measure)

    # This is for the individual RSA, not good right now
    correlation_rsa = []
    for i in range(latent_dim):
        x_i = x[:, i].reshape(-1, 1)
        s_i = compute_similarity(x_i, x_i, similarity_measure)

        x_i_noise = x_noise[:, i].reshape(-1, 1)
        s_i_noise = compute_similarity(x_i_noise, x_i_noise, similarity_measure)

        corr = correlate_rsms(s_i, s_i_noise, rsm_corr)
        correlation_rsa.append(corr)

    correlations_rsa = average_pearson_r(correlation_rsa)
    w = model.fit_transform(s_noise)
    correlation_nmf = aligned_latent_correlation_by_dimension(x, w)
    return {
        "SNR": noise,
        "RSA": correlations_rsa,
        "SymNMF": correlation_nmf,
        "Seed": seed,
        "N": max_obj,
        "K": latent_dim,
    }


def run_hypothesis_test(
    model,
    noise_levels,
    max_objects_list,
    latent_dims_list,
    num_seeds=5,
    similarity_measure="cosine",
    rsm_corr="spearman",
):
    """Run the full simulation with maximum parallelization."""
    # Create all parameter combinations
    param_combinations = [
        (max_obj, latent_dim, noise, seed)
        for max_obj in max_objects_list
        for latent_dim in latent_dims_list
        for noise in noise_levels
        for seed in range(num_seeds)
    ]

    # Run all combinations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_rsa_hypothesis)(model, params, similarity_measure, rsm_corr)
        for params in param_combinations
    )

    return pd.DataFrame(results)


if __name__ == "__main__":

    model = lambda x: x

    noise_levels = np.linspace(0.0, 1.0, 10)
    seeds = range(10)
    max_objects_list = [100]
    latent_dims_list = [10]
    num_seeds = 10

    similarity_measure = "cosine"
    rsm_corr = "spearman"

    spose_embedding = load_spose_embedding(100, 10)

    from ..simulation import SimulationParams

    simulation_params = SimulationParams(
        k=10,
        n_samples=100,
        n_features=100,
        n_components=10,
        snr=1.0,
    )

    # Run the model comparison simulation, eg NMF vs SyNMF
    df = run_model_comparison_simulation(
        model=model,
        simulation_params=simulation_params,
        seeds=seeds,
        noise_levels=noise_levels,
        similarity_measure=similarity_measure,
    )

    df = run_hypothesis_test(
        noise_levels=noise_levels,
        max_objects_list=max_objects_list,
        latent_dims_list=latent_dims_list,
        similarity_measure=similarity_measure,
        rsm_corr=rsm_corr,
        num_seeds=num_seeds,
    )

    df = run_spose_reconstruction_simulation(
        model=model,
        spose_embedding=spose_embedding,
        similarity_measure=similarity_measure,
    )
