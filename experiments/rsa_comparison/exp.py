from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cli import ExperimentContext, register_experiment

from .tasks import run_spose_experiment, run_factorial_experiment

Task = Literal["spose", "factorial"]

DEFAULT_FACTORIAL_LEVELS: dict[str, list[str]] = {
    "animacy": ["animate", "inanimate"],
    "size": ["small", "medium", "large"],
    "curvature": ["straight", "curved"],
    "color": ["red", "blue", "green", "yellow"],
}


@dataclass(slots=True)
class Params:
    task: Task = "spose"
    snrs: list[float] | None = None
    n_repeats: int = 100
    n_permutations: int = 1000
    max_jobs: int = 140
    object_counts: list[int] = field(default_factory=lambda: [50, 100, 200])
    num_dims: int = 10
    similarity_metric: str = "linear"


def add_arguments(parser) -> None:
    parser.add_argument("--task", choices=["spose", "factorial"], default="spose")
    parser.add_argument("--snrs", nargs="+", type=float)
    parser.add_argument("--n-repeats", dest="n_repeats", type=int, default=100)
    parser.add_argument("--n-permutations", dest="n_permutations", type=int, default=1000)
    parser.add_argument("--max-jobs", dest="max_jobs", type=int, default=140)
    parser.add_argument("--object-counts", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--num-dims", dest="num_dims", type=int, default=10)
    parser.add_argument("--similarity-metric", dest="similarity_metric", type=str, default="linear")


def _resolve_snrs(task: Task, snrs: list[float] | None) -> list[float]:
    if snrs is not None:
        return snrs
    return (
        np.linspace(0.0, 1.0, 10).tolist()
        if task == "spose"
        else np.linspace(0.0, 0.6, 10).tolist()
    )


@register_experiment(
    name="rsa_comparison",
    params_type=Params,
    add_arguments=add_arguments,
    description="RSA comparison across SPoSE and factorial designs",
)
def run(context: ExperimentContext, params: Params) -> None:
    snrs = _resolve_snrs(params.task, params.snrs)

    if params.task == "factorial":
        results = run_factorial_experiment(
            snrs=snrs,
            n_repeats=params.n_repeats,
            n_permutations=params.n_permutations,
            max_jobs=params.max_jobs,
            similarity_metric=params.similarity_metric,
            levels=DEFAULT_FACTORIAL_LEVELS,
        )
        task_dir = context.task_dir("factorial")
    else:
        results = run_spose_experiment(
            object_counts=params.object_counts,
            num_dims=params.num_dims,
            snrs=snrs,
            n_repeats=params.n_repeats,
            n_permutations=params.n_permutations,
            max_jobs=params.max_jobs,
        )
        tags = [f"dims{params.num_dims}"]
        task_dir = context.task_dir("spose", *tags)

    context.save_csv(results, "results", subdir=task_dir)
    context.logger.info("Saved RSA comparison results to %s", task_dir)
