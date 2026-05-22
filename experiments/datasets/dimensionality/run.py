"""Dimensionality estimation + CV validation across datasets.

Two stages, sharing one config and one outputs/ folder:

    mode=estimate -> pysrf.estimate_rank, writes coherence_estimate.json
    mode=validate -> pysrf.cross_val_score at p*, writes cross_validation.json
    mode=all      -> estimate then validate

Output per dataset:
    outputs/<name>/coherence_estimate.json
    outputs/<name>/cross_validation.json
    outputs/coherence_estimate_summary*.json
    outputs/cross_validation_summary*.json

Usage:
    python -m experiments.datasets.dimensionality.run mode=estimate
    python -m experiments.datasets.dimensionality.run mode=validate
    python -m experiments.datasets.dimensionality.run mode=all only=[mur92,peterson_animals]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ._estimate import run_estimate
from ._overview import save_overview
from ._validate import run_validate

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    _limit_thread_env()
    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = _resolve_n_jobs(cfg.get("n_jobs", None))
    log.info(f"n_jobs={n_jobs}  output_dir={output_dir}")

    mode = str(cfg.get("mode", "estimate")).lower()
    valid = {"estimate", "validate", "all"}
    if mode not in valid:
        raise ValueError(f"mode must be one of {sorted(valid)}; got {mode!r}")

    if mode in ("estimate", "all"):
        run_estimate(cfg, output_dir, n_jobs)
    if mode in ("validate", "all"):
        run_validate(cfg, output_dir, n_jobs)

    if bool(cfg.get("make_overview", False)):
        dataset_names = [d["name"] for d in OmegaConf.to_container(cfg.datasets, resolve=True)]
        only = cfg.get("only", None)
        if only is not None:
            names = {only} if isinstance(only, str) else set(only)
            dataset_names = [n for n in dataset_names if n in names]
        save_overview(output_dir, dataset_names, output_dir / "overview.pdf")
        log.info(f"overview written to {output_dir / 'overview.pdf'}")


def _resolve_n_jobs(value) -> int:
    cpu = os.cpu_count() or 2
    if value is None:
        return max(1, cpu - 1)
    if int(value) <= 0:
        return max(1, cpu + int(value))
    return int(value)


def _limit_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
