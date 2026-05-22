from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from .datasets import dispatch_dataset_builder


def get_rank_grid(ds_cfg: DictConfig) -> list[int]:
    start, stop, step = ds_cfg.rank_range
    return list(range(start, stop + 1, step))


def build_similarity(
    dataset_cfg: DictConfig,
    subject_id: int | None = None,
) -> np.ndarray:
    return dispatch_dataset_builder(dataset_cfg, subject_id)
