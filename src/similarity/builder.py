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
    dataset: str,
    datasets_cfg: DictConfig,
    subject_id: int | None = None,
) -> np.ndarray:
    if dataset not in datasets_cfg:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(datasets_cfg.keys())}")

    ds_cfg = datasets_cfg[dataset]
    ds_type = ds_cfg.type

    if "requires" in ds_cfg and "subject_id" in ds_cfg.requires:
        if subject_id is None:
            raise ValueError(f"subject_id required for {dataset}")

    if ds_type == "neural_rsm":
        ds = load_dataset(dataset)
        return ds.rsm

    if ds_type == "neural_features":
        loader_kwargs = OmegaConf.to_container(ds_cfg.get("loader_kwargs", {}), resolve=True)
        if subject_id is not None:
            loader_kwargs["subject_id"] = subject_id
        ds = load_dataset(dataset, **loader_kwargs)
        if hasattr(ds, "rsm"):
            return ds.rsm
        similarity_fn = ds_cfg.get("similarity_fn", "gaussian_kernel")
        return compute_similarity(ds.data, ds.data, similarity_fn)

    if ds_type == "feature_file":
        features = np.load(Path(ds_cfg.path) / ds_cfg.features_file)
        if "filter_file" in ds_cfg:
            info = pd.read_csv(ds_cfg.filter_file, dtype={ds_cfg.filter_column: str})
            mask = info[ds_cfg.filter_column].str.contains(ds_cfg.filter_pattern)
            features = features[mask.values, :]
        similarity_fn = ds_cfg.get("similarity_fn", "gaussian_kernel")
        return compute_similarity(features, features, similarity_fn)

    if ds_type == "triplet":
        from utils.io import load_triplets
        triplets, _ = load_triplets(Path(ds_cfg.path), number=ds_cfg.triplet_number)
        return compute_similarity_matrix_from_triplets(ds_cfg.n_objects, triplets)

    if ds_type == "graph":
        adjacency = np.load(Path(ds_cfg.path) / ds_cfg.file)
        if ds_cfg.get("normalize", False):
            max_val = np.nanmax(adjacency)
            if max_val > 1.0:
                adjacency = adjacency / max_val
        fill_nan = ds_cfg.get("fill_nan", None)
        if fill_nan is not None:
            adjacency = np.nan_to_num(adjacency, nan=fill_nan)
        return adjacency

    raise ValueError(f"Unknown dataset type: {ds_type}")
