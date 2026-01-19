"""Dataset builders for similarity matrices."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
from tools.rsa import compute_similarity
from utils.helpers import compute_similarity_matrix_from_triplets
from utils.io import load_triplets

log = logging.getLogger(__name__)


def build_neural_rsm(cfg: DictConfig, subject_id: int | None = None) -> np.ndarray:
    """Build RSM from neural data that already has an RSM (e.g. group-level)."""
    path = cfg.get("path")
    ds = load_dataset(cfg.name, root=path)
    return ds.rsm


def build_neural_features(cfg: DictConfig, subject_id: int | None = None) -> np.ndarray:
    """Build RSM from neural features by computing similarity."""
    raw_kwargs = cfg.get("loader_kwargs", {})
    loader_kwargs = OmegaConf.to_container(raw_kwargs, resolve=True) if raw_kwargs else {}
    if subject_id is not None:
        loader_kwargs["subject_id"] = subject_id

    path = cfg.get("path")
    if path:
        loader_kwargs["root"] = path

    ds = load_dataset(cfg.name, **loader_kwargs)

    if hasattr(ds, "rsm") and ds.rsm is not None:
        return ds.rsm

    similarity_fn = cfg.get("similarity_fn", "gaussian_kernel")
    return compute_similarity(ds.data, ds.data, similarity_fn)


def build_feature_file(cfg: DictConfig, subject_id: int | None = None) -> np.ndarray:
    """Build RSM from a raw feature file (.npy)."""
    path = Path(cfg.get("path"))
    features = np.load(path / cfg.features_file)

    if "filter_file" in cfg:
        info = pd.read_csv(cfg.filter_file, dtype={cfg.filter_column: str})
        mask = info[cfg.filter_column].str.contains(cfg.filter_pattern)
        features = features[mask.values, :]

    similarity_fn = cfg.get("similarity_fn", "gaussian_kernel")
    return compute_similarity(features, features, similarity_fn)


def build_triplet(cfg: DictConfig, subject_id: int | None = None) -> np.ndarray:
    """Build RSM from triplet data with Laplace smoothing."""
    path = Path(cfg.get("path"))
    triplets, _ = load_triplets(path, number=cfg.triplet_number)
    alpha = cfg.get("alpha", 1.0)
    return compute_similarity_matrix_from_triplets(cfg.n_objects, triplets, alpha=alpha)


def build_graph(cfg: DictConfig, subject_id: int | None = None) -> np.ndarray:
    """Build RSM from a graph adjacency matrix."""
    path = Path(cfg.get("path"))
    adjacency = np.load(path / cfg.file)

    if cfg.get("normalize", False):
        max_val = np.nanmax(adjacency)
        if max_val > 1.0:
            adjacency = adjacency / max_val

    fill_nan = cfg.get("fill_nan", None)
    if fill_nan is not None:
        adjacency = np.nan_to_num(adjacency, nan=fill_nan)

    return adjacency


def build_word_association(
    cfg: DictConfig, subject_id: int | None = None
) -> np.ndarray:
    """Build PPMI similarity matrix from word association data (SWOW)."""
    ds = load_dataset(
        cfg.name,
        root=cfg.get("path"),
        use_all_responses=cfg.get("use_all_responses", False),
        top_n_words=cfg.get("top_n_words"),
        min_word_length=cfg.get("min_word_length", 1),
        symmetrization=cfg.get("symmetrization", "geometric_mean"),
        bidirectional_only=cfg.get("bidirectional_only", False),
    )
    return ds.rsm


DATASET_HANDLERS: dict[str, Callable] = {
    "neural_rsm": build_neural_rsm,
    "neural_features": build_neural_features,
    "feature_file": build_feature_file,
    "triplet": build_triplet,
    "graph": build_graph,
    "word_association": build_word_association,
}


def dispatch_dataset_builder(
    cfg: DictConfig, subject_id: int | None = None
) -> np.ndarray:
    """
    Dispatch the appropriate builder based on dataset config type.

    Parameters
    ----------
    cfg : DictConfig
        Dataset configuration.
    subject_id : int | None, optional
        Subject ID for neural datasets.

    Returns
    -------
    np.ndarray
        Computed similarity matrix.
    """
    ds_type = cfg.type
    if ds_type not in DATASET_HANDLERS:
        raise ValueError(
            f"Unknown dataset type: {ds_type}. Available: {list(DATASET_HANDLERS.keys())}"
        )

    if "requires" in cfg and "subject_id" in cfg.requires:
        if subject_id is None:
            raise ValueError(f"subject_id required for {cfg.get('name', 'dataset')}")

    log.info(f"Building similarity matrix for dataset type: {ds_type}")
    return DATASET_HANDLERS[ds_type](cfg, subject_id)
