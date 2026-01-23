"""Compare SRF vs VICE on low-data THINGS triplet task.

Runs SRF on the same training partitions used by VICE and compares validation
accuracy. Uses VICE's estimated dimensionality for each training percentage.

Usage:
    ./scripts/submit experiments/things_behavior/lowdata_comparison.py

Outputs:
    lowdata_comparison.csv - Combined SRF and VICE results by partition
    comparison_summary.csv - Aggregated comparison (mean ± std per method/pct)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF

from src.utils.helpers import compute_similarity_matrix_from_triplets

import logging

log = logging.getLogger(__name__)

PARTITION_DIR = Path("/LOCAL/fmahner/similarity-factorization/data/things/partitions")
VICE_MODEL_DIR = Path(
    "/LOCAL/fmahner/similarity-factorization/outputs/experiments/things_behavior/vice_lowdata/models"
)
N_OBJECTS = 1854

VICE_DIMS = {5: 11, 10: 20, 20: 41, 50: 69}


def _load_partition_triplets(pct: int, part: int) -> tuple[np.ndarray, np.ndarray]:
    """Load train/val triplets for a partition."""
    partition_dir = PARTITION_DIR / f"{pct}pct_part{part}"
    train = np.loadtxt(partition_dir / "train_90.txt", dtype=float).astype(int)
    val = np.loadtxt(partition_dir / "test_10.txt", dtype=float).astype(int)
    return train, val


def _compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    """Evaluate odd-one-out accuracy using softmax over dot product similarities.

    Triplet format: [i, j, k] where (i,j) is chosen pair, k is odd-one-out.
    """
    n_correct = 0
    for i, j, k in triplets:
        sims = np.array([
            embedding[i] @ embedding[j],
            embedding[i] @ embedding[k],
            embedding[j] @ embedding[k],
        ])
        probas = np.exp(sims - sims.max())
        probas /= probas.sum()
        if np.argmax(probas) == 0:
            n_correct += 1
    return n_correct / len(triplets)


def _run_srf_single(pct: int, part: int, rank: int) -> dict:
    """Run SRF on a single partition."""
    train_triplets, val_triplets = _load_partition_triplets(pct, part)
    rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train_triplets, alpha=1.0)

    model = SRF(rank=rank, random_state=42, max_outer=100, max_inner=30)
    embedding = model.fit_transform(rsm)

    val_acc = _compute_triplet_accuracy(embedding, val_triplets)
    train_acc = _compute_triplet_accuracy(embedding, train_triplets)

    return {
        "model": "SRF",
        "pct": pct,
        "part": part,
        "rank": rank,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "n_train": len(train_triplets),
        "n_val": len(val_triplets),
    }


def _get_partitions() -> list[tuple[int, int]]:
    """Get all (pct, part) combinations from the partition directory."""
    partitions = []
    for d in PARTITION_DIR.iterdir():
        if d.is_dir() and "pct_part" in d.name:
            parts = d.name.split("pct_part")
            pct = int(parts[0])
            part = int(parts[1])
            partitions.append((pct, part))
    return sorted(partitions)


def _parse_vice_training_log(log_path: Path) -> dict | None:
    """Parse VICE training log to extract final metrics."""
    if not log_path.exists():
        return None

    text = log_path.read_text()
    lines = text.strip().split("\n")

    final_dim = None
    final_val_acc = None
    final_train_acc = None

    for line in reversed(lines):
        if "Dim:" in line and final_dim is None:
            match = re.search(r"Dim:\s*(\d+)", line)
            if match:
                final_dim = int(match.group(1))
        if "Val acc:" in line and final_val_acc is None:
            match = re.search(r"Val acc:\s*([\d.]+)", line)
            if match:
                final_val_acc = float(match.group(1))
        if "Train acc:" in line and final_train_acc is None:
            match = re.search(r"Train acc:\s*([\d.]+)", line)
            if match:
                final_train_acc = float(match.group(1))
        if final_dim and final_val_acc and final_train_acc:
            break

    if final_dim is None or final_val_acc is None:
        return None

    return {
        "dim": final_dim,
        "val_acc": final_val_acc,
        "train_acc": final_train_acc,
    }


def _collect_vice_results() -> pd.DataFrame:
    """Collect results from all VICE low-data runs."""
    records = []

    for model_dir in VICE_MODEL_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        name = model_dir.name
        match = re.match(r"vice_(\d+)pct_part(\d+)", name)
        if not match:
            continue

        pct = int(match.group(1))
        part = int(match.group(2))

        log_paths = list(model_dir.glob("**/training.log"))
        if not log_paths:
            continue
        metrics = _parse_vice_training_log(log_paths[0])

        if metrics:
            records.append({
                "model": "VICE",
                "pct": pct,
                "part": part,
                "rank": metrics["dim"],
                "val_acc": metrics["val_acc"],
                "train_acc": metrics["train_acc"],
            })

    return pd.DataFrame(records)


def run(cfg: DictConfig) -> None:
    """Run SRF vs VICE low-data comparison."""
    output_dir = Path.cwd()
    n_jobs = cfg.common.get("n_jobs", 16)

    partitions = _get_partitions()
    log.info(f"Found {len(partitions)} partitions")

    # Run SRF on all partitions
    log.info("Running SRF on all partitions...")
    tasks = [(pct, part, VICE_DIMS[pct]) for pct, part in partitions]

    srf_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_srf_single)(pct, part, rank) for pct, part, rank in tasks
    )
    srf_df = pd.DataFrame(srf_results)

    # Collect VICE results
    log.info("Collecting VICE results...")
    vice_df = _collect_vice_results()

    # Combine results
    combined_df = pd.concat([srf_df, vice_df], ignore_index=True)
    combined_df.to_csv(output_dir / "lowdata_comparison.csv", index=False)

    # Create summary
    summary = combined_df.groupby(["model", "pct"]).agg({
        "val_acc": ["mean", "std", "count"],
        "train_acc": ["mean", "std"],
    }).reset_index()
    summary.columns = [
        "model", "pct", "val_acc_mean", "val_acc_std", "n_runs",
        "train_acc_mean", "train_acc_std"
    ]
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    log.info("\nComparison Summary:")
    for pct in sorted(combined_df["pct"].unique()):
        srf_acc = combined_df[(combined_df["model"] == "SRF") & (combined_df["pct"] == pct)]["val_acc"]
        vice_acc = combined_df[(combined_df["model"] == "VICE") & (combined_df["pct"] == pct)]["val_acc"]
        log.info(
            f"  {pct}%: SRF {srf_acc.mean():.3f}±{srf_acc.std():.3f} vs "
            f"VICE {vice_acc.mean():.3f}±{vice_acc.std():.3f}"
        )

    # Copy to data directory for plotting
    data_dir = Path(cfg.project_root) / "outputs/experiments/things_behavior/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(data_dir / "lowdata_comparison.csv", index=False)
    summary.to_csv(data_dir / "lowdata_comparison_summary.csv", index=False)

    log.info(f"\nSaved to {output_dir} and {data_dir}")
