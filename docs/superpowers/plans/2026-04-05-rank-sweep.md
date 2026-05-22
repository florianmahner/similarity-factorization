# Rank Sweep Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `experiments/analyses/things_behavior/rank_sweep/run.py` that computes triplet prediction accuracy vs SRF rank (alpha=0 and alpha=1) using canonical validation triplets, with SPoSE/VICE baselines.

**Architecture:** Single `run.py` using Hydra config, reusing `resources.py` for canonical data loading and `common.py` for SRF fitting and accuracy computation. All (alpha, rank, seed) combinations parallelized via joblib. Outputs a single CSV.

**Tech Stack:** numpy, pandas, joblib, pysrf, omegaconf/hydra

---

### Task 1: Create config

**Files:**
- Create: `configs/experiment/things_behavior/rank_sweep.yaml`

- [ ] **Step 1: Create config file**

```yaml
# THINGS Behavior: Triplet accuracy vs SRF rank

defaults:
  - /base
  - _self_

experiment_name: things_behavior
task: rank_sweep

# === Parameters ===
n_items: 1854
triplet_version: "4.7mio"
dims: 66
alphas: [0, 1]
ranks: [5, 10, 15, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 66]
seeds: [0, 1, 2, 3, 4]

# === Paths ===
dataset_path: ${paths.data_dir}/things
images_path: /SSD/datasets/things/behav1854
vice_embedding_path: ${paths.data_dir}/things/vice_embedding_66d.txt
```

- [ ] **Step 2: Commit**

```bash
git add configs/experiment/things_behavior/rank_sweep.yaml
git commit -m "feat: add rank_sweep config for THINGS accuracy vs rank"
```

---

### Task 2: Create run.py

**Files:**
- Create: `experiments/analyses/things_behavior/rank_sweep/__init__.py`
- Create: `experiments/analyses/things_behavior/rank_sweep/run.py`

- [ ] **Step 1: Create empty `__init__.py`**

```python
```

- [ ] **Step 2: Create `run.py`**

```python
"""Triplet prediction accuracy as a function of SRF rank.

Fits SRF at each rank in a grid for alpha=0 and alpha=1, evaluates on the
canonical THINGS validation triplets. SPoSE and VICE baselines evaluated once
on the same set.

Outputs:
    results.csv -- model, alpha, rank, seed, val_acc
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

from ..common import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
    fit_srf_model,
)
from ..resources import load_resources

log = logging.getLogger(__name__)


def _run_single(
    rsm: np.ndarray,
    val: np.ndarray,
    alpha: int,
    rank: int,
    seed: int,
) -> dict:
    embedding = fit_srf_model(rsm, rank=rank, seed=seed)
    acc = compute_triplet_prediction_accuracy(embedding, val)
    return {
        "model": "SRF",
        "alpha": alpha,
        "rank": rank,
        "seed": seed,
        "val_acc": acc,
    }


def run(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    resources = load_resources(cfg)
    val = resources.validation_triplets

    log.info("Building RSMs (alpha=0 and alpha=1)...")
    rsms = {}
    for alpha in cfg.alphas:
        rsms[alpha] = compute_similarity_matrix_from_triplets(
            cfg.n_items, resources.train_triplets, alpha=alpha
        )

    tasks = [
        (rsms[alpha], val, alpha, rank, seed)
        for alpha in cfg.alphas
        for rank in cfg.ranks
        for seed in cfg.seeds
    ]
    log.info("Running %d SRF fits (%d alphas x %d ranks x %d seeds)...",
             len(tasks), len(cfg.alphas), len(cfg.ranks), len(cfg.seeds))

    records = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_single)(rsm, v, a, r, s) for rsm, v, a, r, s in tasks
    )

    acc_spose = compute_triplet_prediction_accuracy(
        resources.spose_embedding, val
    )
    acc_vice = compute_triplet_prediction_accuracy(
        resources.vice_embedding, val
    )
    records.append({
        "model": "SPoSE",
        "alpha": np.nan,
        "rank": resources.spose_embedding.shape[1],
        "seed": 0,
        "val_acc": acc_spose,
    })
    records.append({
        "model": "VICE",
        "alpha": np.nan,
        "rank": resources.vice_embedding.shape[1],
        "seed": 0,
        "val_acc": acc_vice,
    })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "results.csv", index=False)

    log.info("SPoSE (k=%d): %.4f", resources.spose_embedding.shape[1], acc_spose)
    log.info("VICE  (k=%d): %.4f", resources.vice_embedding.shape[1], acc_vice)

    for alpha in cfg.alphas:
        sub = df[(df["model"] == "SRF") & (df["alpha"] == alpha)]
        summary = sub.groupby("rank")["val_acc"].agg(["mean", "std"])
        log.info("\nalpha=%d:", alpha)
        for rank, row in summary.iterrows():
            log.info("  k=%3d: %.4f +/- %.4f", rank, row["mean"], row["std"])

    log.info("Saved %d rows to %s", len(df), output_dir / "results.csv")
```

- [ ] **Step 3: Commit**

```bash
git add experiments/analyses/things_behavior/rank_sweep/__init__.py \
       experiments/analyses/things_behavior/rank_sweep/run.py
git commit -m "feat: add rank_sweep analysis for THINGS accuracy vs rank"
```

---

### Task 3: Run the analysis

- [ ] **Step 1: Submit the job**

```bash
./scripts/submit experiments/analyses/things_behavior/rank_sweep/run.py experiment=things_behavior/rank_sweep --bg
```

- [ ] **Step 2: Monitor via dashboard or jobs CLI**

```bash
./scripts/jobs rank_sweep --tail
```

- [ ] **Step 3: Verify output**

Check that `experiments/analyses/things_behavior/rank_sweep/outputs/results.csv` exists and has the expected structure:
- 162 rows (160 SRF + 1 SPoSE + 1 VICE)
- Columns: model, alpha, rank, seed, val_acc
- SRF val_acc values should be in the 0.63-0.65 range (canonical validation set, comparable to similarity_48 numbers)
- SPoSE should be ~0.6412, VICE ~0.6422 (matching similarity_48 baselines)
