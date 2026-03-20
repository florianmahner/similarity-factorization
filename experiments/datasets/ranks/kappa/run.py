"""Coherence-based rank estimation across multiple datasets.

Estimates dimensionality (k*) via eigenspace coherence using three methods:
  A) Activation counting -- dimensions whose CI exceeds null threshold
  B) Kappa changepoint -- largest jump in scaled leakage rate
  C) Cluster consensus -- boundary of first signal cluster

Processes datasets sequentially (each saturates all cores internally).
Saves per-dataset JSON (k* estimates) and NPZ (raw coherence arrays).

Usage:
    ./scripts/submit experiments/datasets/ranks/kappa/run.py --bg
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from similarity import build_similarity
from src.coherence import (
    _estimate_kappa_hat,
    analyze_cluster_consensus_across_p,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from utils.helpers import compute_similarity_matrix_from_triplets
from utils.io import load_triplets

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset config loading
# ---------------------------------------------------------------------------


N_OBJECTS = 1854


def _load_dataset_cfg(config_name: str, project_root: Path) -> DictConfig:
    """Load a dataset yaml and resolve path interpolations."""
    cfg_path = project_root / "configs" / "dataset" / f"{config_name}.yaml"
    raw = OmegaConf.load(cfg_path)

    parent = OmegaConf.create({
        "paths": {"data_dir": str(project_root / "data")},
        "dataset": raw,
    })
    OmegaConf.resolve(parent)
    return parent.dataset


def _load_things_triplet_rsm(
    project_root: Path, pct: int, partition: int = 0,
) -> np.ndarray:
    """Load THINGS behavioral triplets at a given percentage split."""
    data_dir = project_root / "data" / "things"
    if pct == 100:
        triplets, _ = load_triplets(data_dir)
    else:
        path = data_dir / "partitions" / f"{pct}pct_part{partition}" / "train_90.txt"
        triplets = np.loadtxt(path).astype(int)
    return compute_similarity_matrix_from_triplets(N_OBJECTS, triplets, alpha=0)


# ---------------------------------------------------------------------------
# k* extraction methods
# ---------------------------------------------------------------------------


def _extract_activation(result: dict) -> dict:
    """Method A: null-calibrated activation counting.

    k* = last dimension whose CI lower bound exceeds null threshold (tau).
    """
    diag = result["diagnostics"]
    tau_kp = diag["tau_kp"]
    x_ci_lo = diag["x_ci_lo"]
    k_list = result["k_list"]

    if tau_kp is None:
        return {"k_star": 0, "method": "activation", "n_activated": 0}

    above = x_ci_lo >= tau_kp
    act_p = np.full(len(k_list), np.nan)
    p_list = result["p"]
    for kk in range(len(k_list)):
        idxs = np.where(above[kk, :])[0]
        if len(idxs) > 0:
            act_p[kk] = float(p_list[idxs[0]])

    valid = ~np.isnan(act_p)
    if np.any(valid):
        k_star = int(k_list[np.where(valid)[0][-1]])
    else:
        k_star = 0

    return {
        "k_star": k_star,
        "method": "activation",
        "n_activated": int(valid.sum()),
        "activation_p": act_p.tolist(),
    }


def _extract_kappa(result: dict) -> dict:
    """Method B: kappa changepoint.

    k* = dimension before the largest jump in scaled leakage rate.
    """
    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_list = result["k_list"]
    p_list = result["p"]

    kappa, kappa_info = _estimate_kappa_hat(x_median, p_list, hi_band_quantile=0.85)
    k_cut, cp_info = kappa_changepoint(kappa, k_list)

    return {
        "k_star": int(k_cut),
        "method": "kappa",
        "kappa": kappa.tolist(),
    }


def _extract_cluster(result: dict) -> dict:
    """Method C: cluster consensus.

    k* = max k in the first (signal) cluster across p-prefixes.
    """
    k_list = result["k_list"]

    cc = analyze_cluster_consensus_across_p(
        result,
        use_baseline_correction=False,
        aggregation="median",
        metric="spearman",
        sim_threshold=0.85,
        k_window=1,
        min_prefix_points=5,
        persistence_level=0.8,
        require_consecutive=3,
        make_plots=False,
    )

    ref_clusters = cc["ref_clusters"]
    if len(ref_clusters) > 0:
        first_cluster = ref_clusters[0]
        k_star = int(k_list[first_cluster[-1]])
        n_clusters = len(ref_clusters)
    else:
        k_star = 0
        n_clusters = 0

    return {
        "k_star": k_star,
        "method": "cluster",
        "n_clusters": n_clusters,
        "cluster_sizes": [len(c) for c in ref_clusters],
    }


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _process_dataset(
    ds_entry: DictConfig,
    coherence_cfg: DictConfig,
    project_root: Path,
    output_dir: Path,
) -> dict:
    """Load dataset, run coherence, extract k*, save results."""
    name = ds_entry.name
    config_name = ds_entry.config
    subject_id = ds_entry.get("subject_id", None)
    k_max = ds_entry.k_max
    k_step = ds_entry.k_step

    log.info(f"{'='*60}")
    log.info(f"Dataset: {name}")
    log.info(f"{'='*60}")

    t0 = time.time()

    # Load RSM -- special handling for THINGS triplet splits
    triplet_pct = ds_entry.get("triplet_pct", None)
    triplet_partition = ds_entry.get("triplet_partition", 0)

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

    # Configure k_list (ensure k_max < n)
    effective_k_max = min(k_max, n - 1)
    k_list = list(range(1, effective_k_max + 1, k_step))
    if k_list[-1] != effective_k_max:
        k_list.append(effective_k_max)
    p_list = np.linspace(
        coherence_cfg.p_min, coherence_cfg.p_max, coherence_cfg.n_p,
    )

    log.info(f"k_list: {k_list[0]}..{k_list[-1]} ({len(k_list)} values), "
             f"p_list: {p_list[0]:.2f}..{p_list[-1]:.2f} ({len(p_list)} values)")
    log.info(f"B={coherence_cfg.B}, B_null={coherence_cfg.B_null}")

    # Run coherence
    t1 = time.time()
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        similarity,
        k_list=k_list,
        p_list=p_list,
        B=coherence_cfg.B,
        random_state=coherence_cfg.random_state,
        compute_null=True,
        B_null=coherence_cfg.B_null,
        alpha_tau=coherence_cfg.alpha_tau,
        ci_level=coherence_cfg.ci_level,
        use_baseline_correction=False,
        n_jobs=os.cpu_count() - 1,
        show_progress=True,
        visualize=False,
    )
    t_coherence = time.time() - t1
    log.info(f"Coherence computed in {t_coherence:.1f}s")

    # Extract k* via three methods
    act = _extract_activation(result)
    kap = _extract_kappa(result)
    clu = _extract_cluster(result)

    log.info(f"k* activation={act['k_star']}, kappa={kap['k_star']}, cluster={clu['k_star']}")

    # Save NPZ with raw arrays
    npz_path = output_dir / f"{name}.npz"
    np.savez_compressed(
        npz_path,
        Iproj_boot=result["Iproj_boot"],
        Iproj_mean=result["Iproj_mean"],
        evals_ref=result["evals_ref"],
        k_list=result["k_list"],
        p_list=result["p"],
        x_mean=result["diagnostics"]["x_mean"],
        x_median=result["diagnostics"]["x_median"],
        x_ci_lo=result["diagnostics"]["x_ci_lo"],
        x_ci_hi=result["diagnostics"]["x_ci_hi"],
        tau_kp=result["diagnostics"]["tau_kp"]
        if result["diagnostics"]["tau_kp"] is not None
        else np.array([]),
        kappa=np.array(kap.get("kappa", [])),
    )
    log.info(f"Saved {npz_path}")

    # Build JSON result
    t_total = time.time() - t0
    record = {
        "dataset": name,
        "n": int(n),
        "k_star_activation": act["k_star"],
        "k_star_kappa": kap["k_star"],
        "k_star_cluster": clu["k_star"],
        "methods": {
            "activation": {
                "k_star": act["k_star"],
                "n_activated": act.get("n_activated", 0),
            },
            "kappa": {
                "k_star": kap["k_star"],
            },
            "cluster": {
                "k_star": clu["k_star"],
                "n_clusters": clu.get("n_clusters", 0),
                "cluster_sizes": clu.get("cluster_sizes", []),
            },
        },
        "params": {
            "k_list": [int(k) for k in k_list],
            "n_p": int(coherence_cfg.n_p),
            "p_min": float(coherence_cfg.p_min),
            "p_max": float(coherence_cfg.p_max),
            "B": int(coherence_cfg.B),
            "B_null": int(coherence_cfg.B_null),
            "alpha_tau": float(coherence_cfg.alpha_tau),
            "ci_level": float(coherence_cfg.ci_level),
        },
        "runtime": {
            "load_sec": round(t_load, 1),
            "coherence_sec": round(t_coherence, 1),
            "total_sec": round(t_total, 1),
        },
    }

    # Save JSON
    json_path = output_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    log.info(f"Saved {json_path}")

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    project_root = Path(cfg.project_root)
    output_dir = Path.cwd()

    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)
    n_datasets = len(datasets)
    log.info(f"Coherence rank estimation for {n_datasets} datasets")

    records = []
    for i, ds_entry in enumerate(datasets):
        ds_cfg = OmegaConf.create(ds_entry)
        log.info(f"\n[{i+1}/{n_datasets}] Processing {ds_cfg.name}...")

        record = _process_dataset(ds_cfg, cfg.coherence, project_root, output_dir)
        records.append(record)

    # Summary table
    df = pd.DataFrame([
        {
            "dataset": r["dataset"],
            "n": r["n"],
            "k_activation": r["k_star_activation"],
            "k_kappa": r["k_star_kappa"],
            "k_cluster": r["k_star_cluster"],
            "runtime_sec": r["runtime"]["total_sec"],
        }
        for r in records
    ])
    df.to_csv(output_dir / "summary.csv", index=False)

    log.info(f"\n{'='*70}")
    log.info("COHERENCE RANK ESTIMATION SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"\n{df.to_string(index=False)}")
    log.info(f"{'='*70}")
    log.info(f"Results saved to {output_dir}")
