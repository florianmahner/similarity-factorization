from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .graph import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    create_models,
    evaluate_model_on_dataset,
    load_all_datasets,
)


def run(cfg: DictConfig) -> None:
    DATASET_REGISTRY.clear()
    MODEL_REGISTRY.clear()
    load_all_datasets()

    seeds = list(cfg.seeds) if cfg.seeds else list(range(10))
    datasets_filter = list(cfg.datasets) if cfg.datasets else None

    tasks = []
    for seed in seeds:
        for dname, ds_info in DATASET_REGISTRY.items():
            if datasets_filter and dname not in datasets_filter:
                continue
            rank = len(np.unique(ds_info["y"]))
            create_models(rank, seed, cfg.max_outer, cfg.max_inner, cfg.verbose)
            tasks.extend(
                (model_info["model"], dname, model_name, seed)
                for model_name, model_info in MODEL_REGISTRY.items()
            )

    results = joblib.Parallel(n_jobs=cfg.n_jobs, verbose=10 if cfg.verbose else 0)(
        joblib.delayed(evaluate_model_on_dataset)(*task) for task in tasks
    )

    df = pd.DataFrame(results)
    df.to_csv(Path.cwd() / "results.csv", index=False)
