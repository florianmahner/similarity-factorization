from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from pysrf import SRF

from tools.rsa import compute_similarity
from utils.helpers import best_pairwise_match
from utils.helpers import add_noise_with_snr

from ..resources import load_resources


def run(cfg: DictConfig) -> None:
    resources = load_resources(cfg)

    # Get parameters from config (set by sweeper)
    snr = cfg.pairwise.snr if "pairwise" in cfg else cfg.snr
    similarity_measure = (
        cfg.pairwise.similarity_measure if "pairwise" in cfg else cfg.similarity_measure
    )
    seed = cfg.seed

    # Add noise and compute similarity
    noisy_spose = add_noise_with_snr(resources.spose_embedding, snr)
    spose_rsm = compute_similarity(
        noisy_spose, resources.spose_embedding, similarity_measure
    )

    # Fit SRF model
    estimator = SRF(
        rank=cfg.get("srf_rank", cfg.dims),
        random_state=seed,
        max_outer=2000,
        max_inner=50,
        tol=1e-4,
        verbose=0,
    )
    w = estimator.fit_transform(spose_rsm)

    # Compute pairwise correlations
    corrs = best_pairwise_match(resources.spose_embedding, w)

    # Save results as JSON (one row per dimension)
    results = [
        {
            "dimension": idx,
            "correlation": float(corr),
            "snr": float(snr),
            "similarity_measure": similarity_measure,
            "seed": int(seed),
        }
        for idx, corr in enumerate(corrs)
    ]

    output_file = Path.cwd() / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
