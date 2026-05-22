"""Migrate the longer-training (max_outer=200) batched CV result into the
canonical dimensionality experiment JSON.

The batched run produced only per-rank summary stats (mean / std / count / sem)
— per-fold scores were not persisted. So the new 5fold block lacks the
``scores`` field; everything else matches the canonical schema produced by
``_record_from_curve``.

The old 5fold block (max_outer=50, n_repeats=20) is preserved as
``cross_validation.outer50.json`` next to the new file.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
CSV_PATH = ROOT / "sandbox/things_behavior/cv_diagnostic/longer_training/outputs/cv_results_batched.csv"
CV_PATH = ROOT / "experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json"
BACKUP_PATH = CV_PATH.with_name("cross_validation.outer50.json")

CV_PROTOCOL_VERSION = "entry_prekfold_diag_v2"
N_FOLDS = 5
N_REPEATS = 10
MAX_OUTER = 200
RANDOM_STATE = 42
N_JOBS = 72
RUNTIME_SEC = 7002.4  # from run_batched.log


def _one_se_rank(ranks, mean, sem, argmin_idx):
    sem_min = sem[argmin_idx]
    if not np.isfinite(sem_min):
        return None
    threshold = mean[argmin_idx] + sem_min
    eligible = [int(r) for r, m in zip(ranks, mean) if np.isfinite(m) and m <= threshold]
    return min(eligible) if eligible else None


def _edge_status(argmin_idx, n_ranks):
    if argmin_idx == 0:
        return "lower_edge"
    if argmin_idx == n_ranks - 1:
        return "upper_edge"
    return "interior"


def main():
    df = pd.read_csv(CSV_PATH).sort_values("rank").reset_index(drop=True)
    ranks = [int(r) for r in df["rank"]]
    mean = df["val_mse_mean"].to_numpy(dtype=float)
    sem = df["val_mse_sem"].to_numpy(dtype=float)
    count = df["count"].to_numpy(dtype=int).tolist()

    argmin_idx = int(np.argmin(mean))
    argmin_rank = ranks[argmin_idx]
    one_se = _one_se_rank(ranks, mean, sem, argmin_idx)
    edge = _edge_status(argmin_idx, len(ranks))

    payload = json.loads(CV_PATH.read_text())
    sampling_fraction = float(payload["sampling_fraction"])

    block = {
        "variant": "5fold",
        "status": "complete",
        "target_ranks": ranks,
        "completed_ranks": ranks,
        "ranks": ranks,
        "argmin_rank": argmin_rank,
        "one_se_rank": one_se,
        "edge_status": edge,
        "val_mse_mean": mean.tolist(),
        "val_mse_sem": sem.tolist(),
        "val_mse_count": count,
        "scores": {},  # per-fold detail not persisted in batched run
        "params": {
            "cv_protocol": CV_PROTOCOL_VERSION,
            "n_folds": N_FOLDS,
            "n_repeats": N_REPEATS,
            "sampling_fraction": sampling_fraction,
            "srf_kwargs": {
                "rho": 3.0,
                "max_inner": 30,
                "tol": 0.0,
                "max_outer": MAX_OUTER,
                "check_input": False,
            },
            "random_state": RANDOM_STATE,
            "rank_strategy": "fixed",
            "rank_spec": {},
            "adaptive": {},
            "n_jobs": N_JOBS,
        },
        "runtime_sec": RUNTIME_SEC,
        "resumed_from_json": False,
        "source": "sandbox/things_behavior/cv_diagnostic/longer_training/run_batched.py",
    }

    if not BACKUP_PATH.exists():
        shutil.copy2(CV_PATH, BACKUP_PATH)
        print(f"backed up old block -> {BACKUP_PATH}")
    else:
        print(f"backup already exists -> {BACKUP_PATH} (not overwriting)")

    payload["validations"]["5fold"] = block
    payload["updated_at_unix"] = time.time()

    tmp = CV_PATH.with_suffix(CV_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(CV_PATH)
    print(f"wrote {CV_PATH}")
    print(f"  argmin_rank={argmin_rank}  one_se_rank={one_se}  edge_status={edge}")
    print(f"  ranks={ranks}")


if __name__ == "__main__":
    main()
