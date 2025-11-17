from __future__ import annotations

"""THINGS behavior experiments orchestrated via the unified CLI.

Each task (pairwise, SPoSE performance, low-data, etc.) shares the same
data-loading pipeline but writes results into its own `task_dir`, making it
easy to inspect individual analyses while reusing shared resources.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pysrf import SRF

from .tasks import (
    low_data_experiment,
    pairwise_reconstruction_experiment,
    run_dimension_reliability_analysis,
    run_spose_dimensionality_analysis,
    spose_48_performance_experiment,
    spose_performance_experiment,
)
from utils.io import load_shared_data, load_triplets

from cli import ExperimentContext, register_experiment


Task = Literal[
    "pairwise",
    "spose_performance",
    "low_data",
    "dimension_reliability",
    "spose_dimensionality",
    "48_performance",
    "all",
]


@dataclass(slots=True)
class Params:
    task: Task = "pairwise"
    dataset_path: Path = Path("data/things")
    images_path: Path = Path("/SSD/fmahner/things")
    vice_embedding_path: Path = Path("data/things/vice_embedding_66d.txt")
    triplet_version: str = "4.7mio"
    dims: int = 66
    n_items: int = 1854
    srf_max_outer: int = 20
    srf_max_inner: int = 50
    srf_rho: float = 1.0
    srf_tol: float = 0.0
    pairwise_seeds: list[int] = field(default_factory=lambda: list(range(10)))
    pairwise_snr_values: list[float] = field(default_factory=lambda: [1.0])
    pairwise_similarity_measures: list[str] = field(default_factory=lambda: ["linear"])
    performance_seeds: list[int] = field(default_factory=lambda: list(range(10)))
    performance48_seeds: list[int] = field(default_factory=lambda: list(range(10)))
    low_data_percentages: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.50, 1.0]
    )
    low_data_seeds: list[int] = field(default_factory=lambda: list(range(20)))
    dimension_reliability_runs: int = 50
    dimension_reliability_jobs: int = -1
    spose_dim_rank_min: int = 5
    spose_dim_rank_max: int = 90
    spose_dim_rank_step: int = 5
    spose_dim_repeats: int = 5
    spose_dim_jobs: int = -1


def add_arguments(parser) -> None:
    parser.add_argument(
        "--task",
        choices=[
            "pairwise",
            "spose_performance",
            "low_data",
            "dimension_reliability",
            "spose_dimensionality",
            "48_performance",
            "all",
        ],
        default="pairwise",
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("data/things"))
    parser.add_argument("--images-path", type=Path, default=Path("/SSD/fmahner/things"))
    parser.add_argument(
        "--vice-embedding-path",
        dest="vice_embedding_path",
        type=Path,
        default=Path("data/things/vice_embedding_66d.txt"),
    )
    parser.add_argument("--triplet-version", type=str, default="4.7mio")
    parser.add_argument("--dims", type=int, default=66)
    parser.add_argument("--n-items", dest="n_items", type=int, default=1854)


@dataclass(slots=True)
class _ThingsResources:
    spose_embedding: np.ndarray
    vice_embedding: np.ndarray
    indices_48: np.ndarray
    rsm_48_true: np.ndarray
    train_triplets: np.ndarray
    validation_triplets: np.ndarray


def _ensure_path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_resources(params: Params) -> _ThingsResources:
    dataset_path = _ensure_path(params.dataset_path)
    images_path = _ensure_path(params.images_path)
    spose_embedding, indices_48, rsm_48_true = load_shared_data(
        dataset_path,
        images_path,
        num_dims=params.dims,
    )
    vice_embedding = np.loadtxt(_ensure_path(params.vice_embedding_path)).astype(
        np.float64
    )
    train_triplets, validation_triplets = load_triplets(
        dataset_path, number=params.triplet_version
    )
    return _ThingsResources(
        spose_embedding=spose_embedding,
        vice_embedding=vice_embedding,
        indices_48=indices_48,
        rsm_48_true=rsm_48_true,
        train_triplets=train_triplets,
        validation_triplets=validation_triplets,
    )


def _srf_params(params: Params) -> dict[str, float | int]:
    return {
        "rank": params.dims,
        "max_outer": params.srf_max_outer,
        "max_inner": params.srf_max_inner,
        "rho": params.srf_rho,
        "tol": params.srf_tol,
    }


# --- Pairwise reconstruction -------------------------------------------------


def _run_pairwise(
    ctx: ExperimentContext, params: Params, resources: _ThingsResources
) -> None:
    ctx.logger.info("Running pairwise reconstruction")
    estimator = SRF(**_srf_params(params))
    df = pairwise_reconstruction_experiment(
        estimator,
        resources.spose_embedding,
        seeds=params.pairwise_seeds,
        snr_values=params.pairwise_snr_values,
        similarity_measures=params.pairwise_similarity_measures,
    )
    tdir = ctx.task_dir("pairwise", f"d{params.dims}")
    ctx.save_csv(df, "pairwise_reconstruction", subdir=tdir)


# --- SPoSE vs SRF performance ------------------------------------------------


def _run_spose_performance(
    ctx: ExperimentContext,
    params: Params,
    resources: _ThingsResources,
) -> None:
    ctx.logger.info("Running SPoSE performance experiment")
    df = spose_performance_experiment(
        resources.spose_embedding,
        resources.vice_embedding,
        resources.indices_48,
        resources.rsm_48_true,
        resources.train_triplets,
        resources.validation_triplets,
        n_items=params.n_items,
        srf_params=_srf_params(params),
        seeds=params.performance_seeds,
    )
    tdir = ctx.task_dir("spose_performance", f"d{params.dims}")
    ctx.save_csv(df, "accuracy_comparison", subdir=tdir)


# --- 48-category analysis ----------------------------------------------------


def _run_48_performance(
    ctx: ExperimentContext,
    params: Params,
    resources: _ThingsResources,
) -> None:
    ctx.logger.info("Running 48 performance experiment")
    df = spose_48_performance_experiment(
        resources.spose_embedding,
        resources.indices_48,
        resources.rsm_48_true,
        resources.train_triplets,
        n_items=params.n_items,
        srf_params=_srf_params(params),
        seeds=params.performance48_seeds,
    )
    tdir = ctx.task_dir("48_performance", f"d{params.dims}")
    ctx.save_csv(df, "performance_48", subdir=tdir)


# --- Low-data sweep ----------------------------------------------------------


def _run_low_data(
    ctx: ExperimentContext,
    params: Params,
    resources: _ThingsResources,
) -> None:
    ctx.logger.info("Running low data experiment")
    df = low_data_experiment(
        resources.train_triplets,
        resources.validation_triplets,
        n_items=params.n_items,
        srf_params=_srf_params(params),
        data_percentages=params.low_data_percentages,
        seeds=params.low_data_seeds,
    )
    tdir = ctx.task_dir("low_data", f"d{params.dims}")
    ctx.save_csv(df, "low_data", subdir=tdir)


# --- Dimension reliability ---------------------------------------------------


def _run_dimension_reliability(
    ctx: ExperimentContext,
    params: Params,
    resources: _ThingsResources,
) -> None:
    ctx.logger.info("Running dimension reliability analysis")
    df = run_dimension_reliability_analysis(
        resources.train_triplets,
        _srf_params(params),
        n_runs=params.dimension_reliability_runs,
        n_jobs=params.dimension_reliability_jobs,
    )
    tdir = ctx.task_dir("dimension_reliability", f"d{params.dims}")
    ctx.save_csv(df, "dimension_reliability", subdir=tdir)


# --- SPoSE dimensionality ----------------------------------------------------


def _run_spose_dimensionality(
    ctx: ExperimentContext,
    params: Params,
    resources: _ThingsResources,
) -> None:
    ctx.logger.info("Running SPoSE dimensionality analysis")
    ranks = range(
        params.spose_dim_rank_min,
        params.spose_dim_rank_max,
        params.spose_dim_rank_step,
    )
    df = run_spose_dimensionality_analysis(
        resources.train_triplets,
        rank_range=ranks,
        n_repeats=params.spose_dim_repeats,
        n_jobs=params.spose_dim_jobs,
    )
    tdir = ctx.task_dir("spose_dimensionality", f"d{params.dims}")
    ctx.save_csv(df, "spose_cross_validation", subdir=tdir)


_TASK_SEQUENCE: list[str] = [
    "pairwise",
    "spose_performance",
    "48_performance",
    "low_data",
    "dimension_reliability",
    "spose_dimensionality",
]

_TaskFn = Callable[[ExperimentContext, Params, _ThingsResources], None]

_TASKS: dict[str, _TaskFn] = {
    "pairwise": _run_pairwise,
    "spose_performance": _run_spose_performance,
    "48_performance": _run_48_performance,
    "low_data": _run_low_data,
    "dimension_reliability": _run_dimension_reliability,
    "spose_dimensionality": _run_spose_dimensionality,
}


@register_experiment(
    name="things_behavior",
    params_type=Params,
    add_arguments=add_arguments,
    description="THINGS experiments with unified CLI",
)
def run(context: ExperimentContext, params: Params) -> None:
    tasks = _TASK_SEQUENCE if params.task == "all" else [params.task]
    context.logger.info(
        "Starting THINGS experiment: %s (tasks=%s)", params.task, ",".join(tasks)
    )
    resources = _load_resources(params)
    for task_name in tasks:
        handler = _TASKS.get(task_name)
        if handler is None:
            raise ValueError(f"Unknown task: {task_name}")
        handler(context, params, resources)
