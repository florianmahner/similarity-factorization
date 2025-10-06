# %%
from models import ADMM
from pysrf import cross_val_score, fit_and_score, mask_missing_entries
import seaborn as sns
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pysrf.bounds import (
    estimate_p_bound,
    estimate_p_bound_fast,
    estimate_p_bound_ultra,
)
from utils.io import load_shared_data
from pathlib import Path

# %%

# Setup paths and results directory
THINGS_DATASET_PATH = Path("/LOCAL/fmahner/similarity-factorization/data/things")
THINGS_IMAGES_PATH = Path("/SSD/datasets/things")
DIMS = 49
RESULTS_DIR = Path(f"/LOCAL/fmahner/similarity-factorization/results/things")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load common data
spose_embedding, indices_48, rsm_48_true = load_shared_data(
    THINGS_DATASET_PATH,
    THINGS_IMAGES_PATH,
    num_dims=DIMS,
)

# things_rsm = spose_embedding @ spose_embedding.

# compute cosine similarity
from tools.rsa import compute_similarity

things_rsm = compute_similarity(spose_embedding, spose_embedding, "cosine")

rsm = things_rsm
# %%
dataset = load_dataset("peterson-various")

# rsm = dataset.group_rsm
rsm = dataset.rsm

# %%
p_min, p_max, S_noise = estimate_p_bound_fast(rsm, random_state=0)

avg = np.mean([p_min, p_max])
print(avg)


# %%

# avg = np.mean([p_min, p_max])
avg = 0.24  # this is the average i get for spose!!
scorer = cross_val_score(
    rsm,
    sampling_fraction=avg,
    n_jobs=140,
    n_repeats=5,
    param_grid={"rank": np.arange(10, 65, 5)},
)


sns.lineplot(scorer.cv_results_, x="rank", y="score")

# %%

best_rank = scorer.best_params_["rank"]
print(best_rank)
# %%


# %%
def evaluate_single_rank(
    sampling_fraction: float,
    trial_id: int,
    rank: int,
    seed: int = 0,
):
    """Evaluate a single rank for a single condition - fully parallelizable."""

    rng = np.random.default_rng(seed)

    s_matrix = rsm

    # Generate validation mask
    rng = np.random.default_rng(seed + 1000)
    val_mask = mask_missing_entries(s_matrix, observed_fraction, rng)

    estimator = SRF(random_state=seed + 2000)
    params = {
        "rank": rank,
        "max_outer": 300,
        "max_inner": 100,
        "rho": 2.0,
        "tol": 0.0,
    }

    # Fit and score
    result = fit_and_score(estimator, s_matrix, val_mask, params, split_idx=0)

    return {
        "sampling_fraction": observed_fraction,
        "trial_id": trial_id,
        "rank": rank,
        "score": result["score"],
        "seed": seed,
    }


fractions = np.linspace(0.01, 1.0, 100)
ranks = np.arange(1, 30)
n_trials = 3
n_jobs = -1

print(f"Running {len(fractions) * len(ranks) * n_trials} trials")

results = Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(evaluate_single_rank)(fraction, trial_id, rank)
    for fraction in fractions
    for rank in ranks
    for trial_id in range(n_trials)
)


# %%
