from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from analyses.clustering.graph import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    create_models,
    evaluate_model_on_dataset,
    load_all_datasets,
)
from cli import ExperimentContext, register_experiment


@dataclass(slots=True)
class Params:
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    datasets: list[str] | None = None
    max_outer: int = 100
    max_inner: int = 100
    verbose: bool = False
    n_jobs: int = -1


def add_arguments(parser) -> None:
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--datasets", nargs="+", type=str)
    parser.add_argument("--max-outer", dest="max_outer", type=int, default=100)
    parser.add_argument("--max-inner", dest="max_inner", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=-1)


def _prepare_tasks(params: Params) -> list[tuple]:
    DATASET_REGISTRY.clear()
    MODEL_REGISTRY.clear()
    load_all_datasets()

    tasks = []
    seeds = params.seeds if params.seeds is not None else list(range(10))
    for seed in seeds:
        for dname, ds_info in DATASET_REGISTRY.items():
            if params.datasets and dname not in params.datasets:
                continue
            rank = len(np.unique(ds_info["y"]))
            create_models(rank, seed, params.max_outer, params.max_inner, params.verbose)
            tasks.extend(
                (
                    model_info["model"],
                    dname,
                    model_name,
                    seed,
                )
                for model_name, model_info in MODEL_REGISTRY.items()
            )
    return tasks


@register_experiment(
    name="graph_clustering",
    params_type=Params,
    add_arguments=add_arguments,
    description="Benchmark clustering models across datasets",
)
def run(context: ExperimentContext, params: Params) -> None:
    tasks = _prepare_tasks(params)
    context.logger.info("Evaluating %d model/dataset combinations", len(tasks))
    results = joblib.Parallel(n_jobs=params.n_jobs, verbose=10 if params.verbose else 0)(
        joblib.delayed(evaluate_model_on_dataset)(*task) for task in tasks
    )

    df = pd.DataFrame(results)
    task_dir = context.task_dir("graph_clustering")
    context.save_csv(df, "results", subdir=task_dir)
    context.logger.info("Benchmark completed; results stored at %s", task_dir)

