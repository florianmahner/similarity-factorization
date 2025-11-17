from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cli import ExperimentContext, register_experiment

from .tasks import run_link_prediction, run_corum_validation

Task = Literal["link_prediction", "corum_validation"]


@dataclass(slots=True)
class Params:
    task: Task = "link_prediction"
    dataset: str | None = None
    splits_dir: Path = Path("data/ppi/splits")
    n_folds: int = 10
    rank: int = 10
    seed: int = 0
    n_jobs: int = -1
    methods: list[str] = field(
        default_factory=lambda: ["srf", "cn", "aa", "ra", "jc", "skipgnn"]
    )
    skipgnn_epochs: int = 200
    seal_hop: int = 1
    seal_epochs: int = 50
    string_data: Path | None = None


def add_arguments(parser) -> None:
    parser.add_argument(
        "--task",
        choices=["link_prediction", "corum_validation"],
        default="link_prediction",
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--splits-dir", dest="splits_dir", type=Path, default=Path("data/ppi/splits")
    )
    parser.add_argument("--n-folds", dest="n_folds", type=int, default=10)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", type=str)
    parser.add_argument(
        "--skipgnn-epochs", dest="skipgnn_epochs", type=int, default=200
    )
    parser.add_argument("--seal-hop", dest="seal_hop", type=int, default=1)
    parser.add_argument("--seal-epochs", dest="seal_epochs", type=int, default=50)
    parser.add_argument("--string-data", dest="string_data", type=Path)


@register_experiment(
    name="ppi",
    params_type=Params,
    add_arguments=add_arguments,
    description="PPI link prediction and CORUM validation",
)
def run(context: ExperimentContext, params: Params) -> None:
    if params.task == "link_prediction":
        if not params.dataset:
            raise ValueError("--dataset required for link_prediction")
        results = run_link_prediction(
            dataset=params.dataset,
            splits_dir=params.splits_dir,
            rank=params.rank,
            seed=params.seed,
            n_folds=params.n_folds,
            methods=params.methods,
            skipgnn_epochs=params.skipgnn_epochs,
            seal_hop=params.seal_hop,
            seal_epochs=params.seal_epochs,
            n_jobs=params.n_jobs,
        )
        task_dir = context.task_dir("link_prediction", params.dataset)
        context.save_csv(results, "results", subdir=task_dir)
    else:
        if not params.string_data:
            raise ValueError("--string-data required for corum_validation")
        embedding, proteins = run_corum_validation(
            params.string_data,
            rank=params.rank,
            seed=params.seed,
        )
        task_dir = context.task_dir("corum_validation")
        np.save(task_dir / "embedding.npy", embedding)
        pd.Series(proteins).to_csv(task_dir / "proteins.txt", index=False, header=False)
    context.logger.info("PPI task %s completed", params.task)
