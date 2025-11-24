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
    rank: int,
    seed: int | None = None,
) -> np.ndarray:
    local_params = {
        "rank": rank,
        "random_state": seed,
        "max_outer": 2000,
        "max_inner": 50,
        "tol": 1e-4,
        "verbose": 0,
    }
    model = SRF(**local_params)
    return model.fit_transform(similarity)
