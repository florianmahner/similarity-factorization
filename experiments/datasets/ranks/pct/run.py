"""Permutation Coherence Test (PCT) for rank estimation across datasets.

Complements the kappa-based kappa.py. For each dataset, runs
the permutation coherence test and saves results as JSON.

Usage:
    ./scripts/submit experiments/datasets/ranks/pct/run.py --bg
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from similarity import build_similarity
from src.coherence import permutation_coherence_test
from utils.helpers import compute_similarity_matrix_from_triplets
from utils.io import load_triplets

log = logging.getLogger(__name__)

N_OBJECTS = 1854


def _load_dataset_cfg(config_name: str, project_root: Path) -> DictConfig:
    raw = OmegaConf.load(project_root / "configs" / "dataset" / f"{config_name}.yaml")
    parent = OmegaConf.create({
        "paths": {"data_dir": str(project_root / "data")},
        "dataset": raw,
    })
    OmegaConf.resolve(parent)
    return parent.dataset


def _load_things_triplet_rsm(project_root: Path, pct: int, partition: int = 0) -> np.ndarray:
    data_dir = project_root / "data" / "things"
    if pct == 100:
        triplets, _ = load_triplets(data_dir)
    else:
        path = data_dir / "partitions" / f"{pct}pct_part{partition}" / "train_90.txt"
        triplets = np.loadtxt(path).astype(int)
    return compute_similarity_matrix_from_triplets(N_OBJECTS, triplets, alpha=0)


def _process_dataset(
    ds_entry: DictConfig,
    pct_cfg: DictConfig,
    project_root: Path,
    output_dir: Path,
) -> dict:
    name = ds_entry.name
    config_name = ds_entry.config
    subject_id = ds_entry.get("subject_id", None)
    k_max = ds_entry.k_max
    triplet_pct = ds_entry.get("triplet_pct", None)
    triplet_partition = ds_entry.get("triplet_partition", 0)

    log.info(f"{'='*60}")
    log.info(f"Dataset: {name}")
    log.info(f"{'='*60}")

    t0 = time.time()
    if triplet_pct is not None:
        log.info(f"Loading THINGS triplets at {triplet_pct}% (partition {triplet_partition})...")
        similarity = _load_things_triplet_rsm(project_root, triplet_pct, triplet_partition)
    else:
        dataset_cfg = _load_dataset_cfg(config_name, project_root)
        log.info(f"Loading RSM for {name}...")
        similarity = build_similarity(dataset_cfg, subject_id=subject_id)

    n = similarity.shape[0]
    t_load = time.time() - t0
    log.info(f"RSM: {similarity.shape}, load time: {t_load:.1f}s")

    effective_k_max = min(k_max, n - 1)
    p_list = np.linspace(
        pct_cfg.p_min, pct_cfg.p_max, pct_cfg.n_p,
    )

    center = ds_entry.get("center", False)
    log.info(f"k_max={effective_k_max}, center={center}, "
             f"p_list: {p_list[0]:.2f}..{p_list[-1]:.2f} ({len(p_list)} values)")
    log.info(f"B={pct_cfg.B}, J={pct_cfg.J}")

    t1 = time.time()
    result = permutation_coherence_test(
        similarity,
        k_max=effective_k_max,
        p_list=p_list,
        B=pct_cfg.B,
        J=pct_cfg.J,
        alpha=pct_cfg.alpha,
        center=center,
        random_state=pct_cfg.random_state,
        show_progress=True,
    )
    t_pct = time.time() - t1

    pv = result["pvalues"]
    n_sig = int(np.sum(pv < pct_cfg.alpha))
    log.info(f"PCT: k*={result['k_star']}, n_significant={n_sig}, time={t_pct:.1f}s")

    # Save NPZ
    np.savez_compressed(
        output_dir / f"{name}.npz",
        iproj_observed=result["iproj_observed"],
        iproj_null=result["iproj_null"],
        thresholds=result["thresholds"],
        evals_ref=result["evals_ref"],
        pvalues=result["pvalues"],
        p_list=result["p_list"],
    )

    # Save JSON
    t_total = time.time() - t0
    record = {
        "dataset": name,
        "n": int(n),
        "k_star_pct": result["k_star"],
        "n_significant": n_sig,
        "pvalues": pv.tolist(),
        "params": {
            "k_max": effective_k_max,
            "n_p": int(pct_cfg.n_p),
            "p_min": float(pct_cfg.p_min),
            "p_max": float(pct_cfg.p_max),
            "B": int(pct_cfg.B),
            "J": int(pct_cfg.J),
            "alpha": float(pct_cfg.alpha),
        },
        "runtime": {
            "load_sec": round(t_load, 1),
            "pct_sec": round(t_pct, 1),
            "total_sec": round(t_total, 1),
        },
    }
    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(record, f, indent=2)
    log.info(f"Saved {name}.json + .npz")

    return record


def run(cfg: DictConfig) -> None:
    project_root = Path(cfg.project_root)
    output_dir = Path.cwd()

    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)
    n_datasets = len(datasets)
    log.info(f"PCT rank estimation for {n_datasets} datasets")

    records = []
    for i, ds_entry in enumerate(datasets):
        ds_cfg = OmegaConf.create(ds_entry)
        log.info(f"\n[{i+1}/{n_datasets}] Processing {ds_cfg.name}...")
        record = _process_dataset(ds_cfg, cfg.pct, project_root, output_dir)
        records.append(record)

    # Summary
    log.info(f"\n{'='*70}")
    log.info("PCT RANK ESTIMATION SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'dataset':>20s}  {'n':>6s}  {'k*_pct':>6s}")
    for r in records:
        log.info(f"{r['dataset']:>20s}  {r['n']:>6d}  {r['k_star_pct']:>6d}")
    log.info(f"{'='*70}")
    log.info(f"Results saved to {output_dir}")
