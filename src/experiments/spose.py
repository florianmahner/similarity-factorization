import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from joblib import Parallel, delayed

from utils.io import load_things_image_data
from models.admm import ADMM
from tools.rsa import correlate_rsms, reconstruct_rsm_batched, compute_similarity
from utils.simulation import add_noise_with_snr
from utils.helpers import best_pairwise_match
from tqdm import tqdm


def process_single_model(model, spose_embedding, rsm, snr, seed):
    np.random.seed(seed)
    w = model.fit_transform(rsm)
    corrs = best_pairwise_match(spose_embedding, w)
    return [
        {"Dimension": i, "Correlation": corr, "SNR": snr, "Seed": seed}
        for i, corr in enumerate(corrs)
    ]


def run_spose_reconstruction_simulation(
    model,
    spose_embedding,
    seeds=[0, 42, 123, 456, 789],
    snr=1.0,
    similarity_measure="cosine",
):
    noisy_spose = (
        spose_embedding if snr == 1.0 else add_noise_with_snr(spose_embedding, snr)
    )
    simk = compute_similarity(noisy_spose, noisy_spose, similarity_measure)

    tasks = [(model, spose_embedding, simk, snr, seed) for seed in seeds]
    all_results = Parallel(n_jobs=-1)(
        delayed(process_single_model)(*task) for task in tqdm(tasks)
    )
    return pd.DataFrame([item for sublist in all_results for item in sublist])


def load_embeddings(embedding_path: Path) -> np.ndarray:
    return np.maximum(np.loadtxt(embedding_path), 0)


def load_concept_mappings(
    words_path: Path, things_path: str, replacements: dict
) -> list:
    words48 = pd.read_csv(words_path)
    cls_names = [replacements.get(name, name) for name in words48["Word"].values]

    images = load_things_image_data(things_path, filter_behavior=True)
    categories = [" ".join(Path(f).stem.split("_")[0:-1]) for f in images]
    return [categories.index(c) for c in cls_names]


def load_ground_truth(ground_truth_path: Path, rsm_key: str) -> np.ndarray:
    return 1 - loadmat(ground_truth_path)[rsm_key]


def sample_triplets(triplets: np.ndarray, percentage: float, seed: int) -> np.ndarray:
    if percentage >= 1.0:
        return triplets
    np.random.seed(seed)
    n_samples = int(len(triplets) * percentage)
    indices = np.random.choice(len(triplets), size=n_samples, replace=False)
    return triplets[indices]


def compute_similarity_matrix(
    n: int, triplets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))

    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            if a != b:
                shown[a, b] += 1
                shown[b, a] += 1

        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1

    mask = shown > 0
    np.fill_diagonal(mask, True)  # we still want self similarities in the loss
    similarity = np.divide(
        counts,
        shown,
        out=np.nan * np.ones_like(counts),
        where=shown != 0,
    )
    np.fill_diagonal(similarity, 1.0)

    return similarity


def fit_admm_model(
    similarity: np.ndarray, params: dict, seed: int = None
) -> np.ndarray:
    local_params = params.copy()
    local_params["random_state"] = seed
    model = ADMM.set_params(**local_params)
    return model.fit_transform(similarity)


def softmax(w_i: np.ndarray, w_j: np.ndarray, w_k: np.ndarray) -> int:
    similarities = np.array([np.dot(w_i, w_j), np.dot(w_i, w_k), np.dot(w_j, w_k)])
    return np.argmax(np.exp(similarities) / np.sum(np.exp(similarities)))


def evaluate_triplets(
    triplets: np.ndarray, admm_embedding: np.ndarray, spose_embedding: np.ndarray
) -> tuple[float, float]:
    acc_admm = acc_spose = 0
    for i, j, k in triplets:
        acc_admm += (
            softmax(admm_embedding[i], admm_embedding[j], admm_embedding[k]) == 0
        )
        acc_spose += (
            softmax(spose_embedding[i], spose_embedding[j], spose_embedding[k]) == 0
        )
    return acc_admm / len(triplets), acc_spose / len(triplets)


def reconstruct_admm_rsm(w):
    # TODO replace with reconstrut function
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
    validation_triplets,
    admm_params,
):
    admm_embedding = fit_admm_model(similarity, admm_params, seed=seed)

    rsm_48_spose = reconstruct_rsm_batched(spose_embedding[indices_48])
    rsm_admm = reconstruct_admm_rsm(admm_embedding)
    rsm_48_admm = rsm_admm[np.ix_(indices_48, indices_48)]

    corr_admm = correlate_rsms(rsm_48_admm, rsm_48_true)
    corr_spose = correlate_rsms(rsm_48_spose, rsm_48_true)
    acc_admm, acc_spose = evaluate_triplets(
        validation_triplets, admm_embedding, spose_embedding
    )

    return [
        {"Model": "ADMM", "Correlation": corr_admm, "Accuracy": acc_admm, "seed": seed},
        {
            "Model": "SPoSE",
            "Correlation": corr_spose,
            "Accuracy": acc_spose,
            "seed": seed,
        },
    ]


def run_low_data_experiment(
    seed, data_percentage, full_triplets, admm_params, validation_triplets
):
    # TODO Check how many triplets are sampled and what the fraction of the similarity matrix is, eg how much is missing.
    sampled_triplets = sample_triplets(full_triplets, data_percentage, seed)

    similarity, mask = compute_similarity_matrix(1854, sampled_triplets)

    if mask.sum() == 0:
        return None

    admm_embedding = fit_admm_model(similarity, mask, admm_params, seed=seed)
    acc_admm = sum(
        softmax(admm_embedding[i], admm_embedding[j], admm_embedding[k]) == 0
        for i, j, k in validation_triplets
    ) / len(validation_triplets)

    return {
        "Model": "ADMM",
        "Data_Percentage": data_percentage,
        "N_Triplets": len(sampled_triplets),
        "Accuracy": acc_admm,
        "seed": seed,
    }


def load_shared_data(
    spose_embedding_path: Path,
    words48_path: Path,
    things_dataset_path: str,
    category_replacements: dict,
    ground_truth_rsm_path: Path,
    ground_truth_rsm_key: str,
    triplets_path: Path,
    validation_triplets_path: Path,
    n_items: int,
) -> tuple:
    spose_embedding = load_embeddings(spose_embedding_path)
    indices_48 = load_concept_mappings(
        words48_path, things_dataset_path, category_replacements
    )
    rsm_48_true = load_ground_truth(ground_truth_rsm_path, ground_truth_rsm_key)

    triplets = np.loadtxt(triplets_path).astype(int)
    similarity = compute_similarity_matrix(n_items, triplets)
    validation_triplets = np.loadtxt(validation_triplets_path).astype(int)

    return (
        spose_embedding,
        indices_48,
        rsm_48_true,
        similarity,
        validation_triplets,
        triplets,
    )
