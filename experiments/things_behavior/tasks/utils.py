from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from pysrf import SRF


def run_experiment(
    trial_func: Callable[..., list[dict]],
    param_grid: dict,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    combinations = list(ParameterGrid(param_grid))
    if verbose:
        print(f"Running {len(combinations)} parameter combinations")

    results = Parallel(n_jobs=n_jobs)(
        delayed(trial_func)(**params)
        for params in tqdm(combinations, desc="Running experiment")
    )
    return pd.DataFrame(itertools.chain.from_iterable(results))


def fit_srf_model(
    similarity: np.ndarray,
    params: dict,
    seed: int | None = None,
) -> np.ndarray:
    local_params = params.copy()
    local_params["random_state"] = seed
    model = SRF(**local_params)
    return model.fit_transform(similarity)


def reconstruct_srf_rsm(embedding: np.ndarray) -> np.ndarray:
    sim = embedding @ embedding.T
    sim_copy = sim.copy()
    np.fill_diagonal(sim_copy, 0)

    min_val, max_val = sim_copy.min(), sim_copy.max()
    normalized = (
        (sim - min_val) / (max_val - min_val) if max_val > min_val else sim.copy()
    )
    np.fill_diagonal(normalized, 1)
    return normalized
