"""Dataset loading shared by estimate.py and validate.py."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from similarity import build_similarity

log = logging.getLogger(__name__)


def load_similarity(
    ds_entry: DictConfig,
    project_root: Path,
    cache_dir: Path | None = None,
    use_cache: bool = False,
    force_cache: bool = False,
) -> np.ndarray:
    """Build the similarity matrix for a dataset entry."""
    if use_cache and cache_dir is not None:
        cache_path = cache_dir / f"{ds_entry.name}.npy"
        if cache_path.exists() and not force_cache:
            log.info(f"  loading cached similarity: {cache_path}")
            return np.load(cache_path)

    dataset_cfg = _load_dataset_cfg(ds_entry.config, project_root)
    _apply_dataset_overrides(dataset_cfg, ds_entry)
    subject_id = ds_entry.get("subject_id", None)
    similarity = build_similarity(dataset_cfg, subject_id=subject_id)

    if use_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, similarity)
        log.info(f"  cached similarity: {cache_path}")
    return similarity


def _load_dataset_cfg(config_name: str, project_root: Path) -> DictConfig:
    """Load and resolve a configs/dataset/<config_name>.yaml file."""
    cfg_path = project_root / "configs" / "dataset" / f"{config_name}.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": OmegaConf.load(project_root / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(project_root),
    })
    OmegaConf.resolve(parent)
    return parent.dataset


def _apply_dataset_overrides(dataset_cfg: DictConfig, ds_entry: DictConfig) -> None:
    """Merge experiment-local loader overrides into the dataset config."""
    if "loader_kwargs" in ds_entry:
        current = dataset_cfg.get("loader_kwargs", {})
        merged = OmegaConf.merge(current, ds_entry.loader_kwargs)
        dataset_cfg.loader_kwargs = merged
