"""Unified bounds estimation for any dataset.

Usage:
    ./scripts/submit experiments/estimate_bounds.py dataset=swow
    ./scripts/submit experiments/estimate_bounds.py dataset=nsd subject_id=1
    ./scripts/submit experiments/estimate_bounds.py dataset=things_behavior
    ./scripts/submit experiments/estimate_bounds.py dataset=things_macaque
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pysrf.bounds import estimate_sampling_bounds_ultra
from similarity import build_similarity

log = logging.getLogger(__name__)


def _prepare_similarity(similarity: np.ndarray) -> np.ndarray:
    """Handle NaN values in similarity matrix for bounds estimation.

    Covers all dataset types:
    - PPMI (word_association): NaN diagonal -> fill with max off-diagonal
    - Triplet (things_behavior): sparse NaN -> fill with 0
    - Neural (nsd, things_macaque): typically clean, no-op
    """
    # Fill NaN diagonal with max off-diagonal value (handles PPMI case)
    diag_vals = np.diag(similarity).copy()
    n_diag_nan = np.isnan(diag_vals).sum()
    if n_diag_nan > 0:
        # Temporarily set diagonal to 0 to compute off-diagonal max
        similarity_copy = similarity.copy()
        np.fill_diagonal(similarity_copy, 0)
        off_diag_max = np.nanmax(similarity_copy)
        np.fill_diagonal(similarity, off_diag_max)
        log.info(
            f"Filled {n_diag_nan} NaN diagonal values with max off-diagonal ({off_diag_max:.4f})"
        )

    # # Replace any remaining NaN with 0 (handles triplet case)
    # FIXME We are not allowed to fit with NaNs.
    # n_remaining_nan = np.isnan(similarity).sum()
    # if n_remaining_nan > 0:
    #     log.info(f"Replacing {n_remaining_nan} remaining NaN values with 0")
    #     similarity = np.nan_to_num(similarity, nan=0.0)

    return similarity


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")

    # Warn if using incomplete subject for NSD
    valid_subjects = cfg.dataset.get("valid_subjects")
    if valid_subjects and subject_id not in valid_subjects:
        log.warning(
            f"Subject {subject_id} did not complete all sessions. "
            f"Valid subjects: {valid_subjects}. Matrix size will differ."
        )

    log.info(f"Loading similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"Similarity matrix shape: {similarity.shape}")

    similarity = _prepare_similarity(similarity)

    start_time = time.time()
    pmin, pmax, _ = estimate_sampling_bounds_ultra(
        similarity,
        random_state=cfg.common.random_state,
        verbose=True,
    )
    elapsed = time.time() - start_time

    # Build metadata from dataset config
    bounds = {
        "dataset": cfg.dataset.name,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "mean_sampling_fraction": float(0.5 * (pmin + pmax)),
        "matrix_shape": list(similarity.shape),
        "random_state": cfg.common.random_state,
        "computation_time_seconds": float(elapsed),
    }

    # Add subject_id if present
    if subject_id is not None:
        bounds["subject_id"] = int(subject_id)

    # Add dataset-specific fields from config
    dataset_fields = [
        "use_all_responses",
        "symmetrization",
        "triplet_number",
        "n_objects",
        "similarity_fn",
    ]
    for field in dataset_fields:
        if field in cfg.dataset:
            bounds[field] = cfg.dataset[field]

    # Add loader_kwargs if present
    if "loader_kwargs" in cfg.dataset:
        bounds["loader_kwargs"] = OmegaConf.to_container(cfg.dataset.loader_kwargs)

    log.info(f"Bounds: pmin={pmin:.4f}, pmax={pmax:.4f}, time={elapsed:.1f}s")

    # Determine output path
    out_dir = Path.cwd()
    if subject_id is not None:
        out_dir = out_dir / f"subj{subject_id:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)

    log.info(f"Saved bounds to {out_dir / 'bounds.json'}")
