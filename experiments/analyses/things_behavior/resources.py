from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from utils.io import load_shared_data, load_triplets


@dataclass(slots=True)
class ThingsResources:
    spose_embedding: np.ndarray
    vice_embedding: np.ndarray
    indices_48: np.ndarray
    rsm_48_true: np.ndarray
    train_triplets: np.ndarray
    validation_triplets: np.ndarray


def load_resources(cfg: DictConfig) -> ThingsResources:
    dataset_path = Path(cfg.dataset_path)
    images_path = Path(cfg.images_path)
    vice_path = Path(cfg.vice_embedding_path)

    spose_embedding, indices_48, rsm_48_true = load_shared_data(
        dataset_path, images_path, num_dims=cfg.dims
    )
    vice_embedding = np.maximum(np.loadtxt(vice_path), 0)
    train_triplets, validation_triplets = load_triplets(
        dataset_path, number=cfg.triplet_version
    )

    return ThingsResources(
        spose_embedding=spose_embedding,
        vice_embedding=vice_embedding,
        indices_48=indices_48,
        rsm_48_true=rsm_48_true,
        train_triplets=train_triplets,
        validation_triplets=validation_triplets,
    )
