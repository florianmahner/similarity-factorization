from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig
from pysrf.bounds import estimate_sampling_bounds_fast

from similarity import build_similarity


def run(cfg: DictConfig) -> None:
    similarity = build_similarity(cfg.dataset, subject_id=cfg.estimate.subject_id)

    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity,
        random_state=cfg.common.random_state,
        n_jobs=cfg.common.n_jobs,
        verbose=False,
    )

    bounds = {
        "dataset": cfg.dataset.get("name", "unknown"),
        "subject_id": cfg.estimate.subject_id,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "mean_sampling_fraction": float(0.5 * (pmin + pmax)),
        "matrix_shape": list(similarity.shape),
        "random_state": cfg.common.random_state,
    }

    # Save to experiment output dir instead of cwd/dataset
    out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)
