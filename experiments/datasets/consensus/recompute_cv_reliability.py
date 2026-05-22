"""Recompute CV split-half dimension reliability from existing runs.npy.

Avoids re-fitting SRF: loads the aligned ensemble (runs.npy) that consensus/run.py
has already saved per dataset, applies the current src/tools/stats.dimension_reliability,
and overwrites cv_reliability.npy + reliability_summary.csv used by
experiments/figures/plot_embeddings/plot.py.

Usage:
    poetry run python experiments/datasets/consensus/recompute_cv_reliability.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tools.stats import dimension_reliability

log = logging.getLogger("recompute_cv_reliability")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CONSENSUS_DIR = Path(__file__).resolve().parent / "outputs"

DATASETS: list[dict] = [
    {"name": "mur92", "subject_id": None},
    {"name": "peterson-animals", "subject_id": None},
    {"name": "peterson-various", "subject_id": None},
    {"name": "vgg16", "subject_id": None},
    {"name": "swow", "subject_id": None},
    {"name": "nsd", "subject_id": 1},
    {"name": "nsd", "subject_id": 2},
]

N_SPLITS = 100
RANDOM_STATE = 42


def _dataset_dir(name: str, subject_id: int | None) -> Path:
    d = CONSENSUS_DIR / name
    if subject_id is not None:
        d = d / f"subj{subject_id:02d}"
    return d


def _key(name: str, subject_id: int | None) -> str:
    if subject_id is not None:
        return f"{name}_subj{subject_id:02d}"
    return name


def _recompute_one(name: str, subject_id: int | None) -> dict | None:
    d = _dataset_dir(name, subject_id)
    runs_path = d / "runs.npy"
    if not runs_path.exists():
        log.warning(f"skip (no runs.npy): {d}")
        return None

    key = _key(name, subject_id)
    runs = np.load(runs_path)
    n_runs, n_objects, n_dims = runs.shape
    log.info(f"== {key}: runs shape={runs.shape} ==")

    cv_rel = dimension_reliability(runs, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    np.save(d / "cv_reliability.npy", cv_rel)
    log.info(
        f"cv_reliability: mean={cv_rel.mean():.4f} "
        f"min={cv_rel.min():.4f} max={cv_rel.max():.4f}"
    )

    naive_path = d / "reliability.npy"
    naive_mean = float(np.load(naive_path).mean()) if naive_path.exists() else float("nan")

    return {
        "dataset": key,
        "rank": int(n_dims),
        "n_runs": int(n_runs),
        "cv_rel_mean": float(cv_rel.mean()),
        "cv_rel_min": float(cv_rel.min()),
        "cv_rel_max": float(cv_rel.max()),
        "naive_rel": naive_mean,
    }


def main() -> None:
    rows: list[dict] = []
    for ds in DATASETS:
        row = _recompute_one(ds["name"], ds["subject_id"])
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    out = CONSENSUS_DIR / "reliability_summary.csv"
    df.to_csv(out, index=False)
    log.info(f"\nWrote {out}")
    log.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
