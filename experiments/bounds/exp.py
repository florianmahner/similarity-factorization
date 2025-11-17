from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from config import get_dataset_path
from datasets import load_dataset
from pysrf.bounds import estimate_sampling_bounds_fast
from tools.rsa import compute_similarity

from cli import ExperimentContext, register_experiment

SMALL_DATASETS: set[str] = {
    "mur92",
    "cichy118",
    "peterson-animals",
    "peterson-various",
}
LARGE_DATASETS: set[str] = {
    "nsd",
    "things-monkey-22k",
    "vit",
    "things-ooo",
    "word-association",
}

DatasetSlug = Literal[
    "mur92",
    "cichy118",
    "peterson-animals",
    "peterson-various",
    "nsd",
    "things-monkey-22k",
    "vit",
    "things-ooo",
    "word-association",
]


@dataclass(slots=True)
class Params:
    dataset: DatasetSlug
    subject_id: int | None = None
    n_jobs: int = -1
    random_state: int = 0


def add_arguments(parser) -> None:
    parser.add_argument("--dataset", choices=sorted(SMALL_DATASETS | LARGE_DATASETS))
    parser.add_argument("--subject-id", dest="subject_id", type=int)
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=-1)
    parser.add_argument("--random-state", dest="random_state", type=int, default=0)


def _load_similarity(dataset_slug: DatasetSlug, subject_id: int | None) -> np.ndarray:
    if dataset_slug == "nsd":
        if subject_id is None:
            raise ValueError("subject_id is required for nsd")
        ds = load_dataset(
            "nsd",
            subject_id=subject_id,
            roi_name="streams",
            space="func1pt8mm",
        )
        return compute_similarity(ds.data, ds.data, "gaussian_kernel")

    if dataset_slug == "things-monkey-22k":
        ds = load_dataset(
            "things-monkey-22k",
            min_reliab=0.3,
            monkey_type="F",
            roi="it",
        )
        if hasattr(ds, "rsm"):
            return ds.rsm
        return compute_similarity(ds.data, ds.data, "gaussian_kernel")

    if dataset_slug == "vit":
        features = np.load(
            "/SSD/projects/deepsim/raw/features/dataset/openai/ViT-L-14/visual/features.npy"
        )
        import pandas as pd

        image_info = pd.read_csv(
            "/SSD/projects/deepsim/raw/features/image_info.csv", dtype={"filename": str}
        )
        plus_indices = image_info[image_info["filename"].str.contains("_plus")].index
        features = features[plus_indices, :]
        return compute_similarity(features, features, "gaussian_kernel")

    if dataset_slug == "things-ooo":
        from utils.io import load_triplets
        from analysis.things.common import compute_similarity_matrix_from_triplets

        train_triplets, _ = load_triplets(
            Path(get_dataset_path("things-ooo")), number="4.7mio"
        )
        similarity = compute_similarity_matrix_from_triplets(1854, train_triplets)
        return np.nan_to_num(similarity, nan=0.0)

    if dataset_slug == "word-association":
        data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/ppi")
        adjacency = np.load(data_dir / "string_adjacency_weighted_nan.npy")
        max_val = np.nanmax(adjacency)
        if max_val > 1.0:
            adjacency = adjacency / max_val
        return np.nan_to_num(adjacency, nan=0.0)

    ds = load_dataset(dataset_slug)
    if hasattr(ds, "rsm"):
        return ds.rsm
    return compute_similarity(ds.data, ds.data, "gaussian_kernel")


def _estimate_bounds(
    similarity: np.ndarray,
    dataset: str,
    subject_id: int | None,
    random_state: int,
    n_jobs: int,
) -> dict:
    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    mean_sampling_fraction = 0.5 * (pmin + pmax)
    return {
        "dataset": dataset,
        "subject_id": subject_id,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "mean_sampling_fraction": float(mean_sampling_fraction),
        "matrix_shape": list(similarity.shape),
        "random_state": random_state,
    }


@register_experiment(
    name="bounds",
    params_type=Params,
    add_arguments=add_arguments,
    description="Sampling bounds for multiple datasets",
)
def run(context: ExperimentContext, params: Params) -> None:
    similarity = _load_similarity(params.dataset, params.subject_id)
    bounds = _estimate_bounds(
        similarity,
        dataset=params.dataset,
        subject_id=params.subject_id,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
    )

    tag = params.dataset
    if params.subject_id is not None:
        tag = f"{tag}/subj{params.subject_id:02d}"
    task_dir = context.task_dir("bounds", tag)
    bounds_path = task_dir / "bounds.json"
    bounds_path.write_text(json.dumps(bounds, indent=2))
    context.logger.info("Saved bounds to %s", bounds_path)

