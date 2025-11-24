from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
from tools.rsa import compute_similarity
from utils.helpers import compute_similarity_matrix_from_triplets


def get_rank_grid(ds_cfg: DictConfig) -> list[int]:
    start, stop, step = ds_cfg.rank_range
    return list(range(start, stop + 1, step))


def build_similarity(
    dataset_cfg: DictConfig,
    subject_id: int | None = None,
) -> np.ndarray:
    ds_type = dataset_cfg.type
    ds_path = dataset_cfg.get("path")  # Path is now optional/derived from config

    if "requires" in dataset_cfg and "subject_id" in dataset_cfg.requires:
        if subject_id is None:
            raise ValueError(
                f"subject_id required for {dataset_cfg.get('name', 'dataset')}"
            )

    if ds_type == "neural_rsm":
        ds = load_dataset(dataset_cfg.name, root=ds_path)
        return ds.rsm

    if ds_type == "neural_features":
        loader_kwargs = OmegaConf.to_container(
            dataset_cfg.get("loader_kwargs", {}), resolve=True
        )
        if subject_id is not None:
            loader_kwargs["subject_id"] = subject_id
        if ds_path:
            loader_kwargs["root"] = ds_path

        ds = load_dataset(dataset_cfg.name, **loader_kwargs)
        if hasattr(ds, "rsm"):
            return ds.rsm
        similarity_fn = dataset_cfg.get("similarity_fn", "gaussian_kernel")
        return compute_similarity(ds.data, ds.data, similarity_fn)

    if ds_type == "feature_file":
        features = np.load(Path(ds_path) / dataset_cfg.features_file)
        if "filter_file" in dataset_cfg:
            info = pd.read_csv(
                dataset_cfg.filter_file, dtype={dataset_cfg.filter_column: str}
            )
            mask = info[dataset_cfg.filter_column].str.contains(
                dataset_cfg.filter_pattern
            )
            features = features[mask.values, :]
        similarity_fn = dataset_cfg.get("similarity_fn", "gaussian_kernel")
        return compute_similarity(features, features, similarity_fn)

    if ds_type == "triplet":
        from utils.io import load_triplets

        triplets, _ = load_triplets(Path(ds_path), number=dataset_cfg.triplet_number)
        return compute_similarity_matrix_from_triplets(dataset_cfg.n_objects, triplets)

    if ds_type == "graph":
        adjacency = np.load(Path(ds_path) / dataset_cfg.file)
        if dataset_cfg.get("normalize", False):
            max_val = np.nanmax(adjacency)
            if max_val > 1.0:
                adjacency = adjacency / max_val
        fill_nan = dataset_cfg.get("fill_nan", None)
        if fill_nan is not None:
            adjacency = np.nan_to_num(adjacency, nan=fill_nan)
        return adjacency

    raise ValueError(f"Unknown dataset type: {ds_type}")
