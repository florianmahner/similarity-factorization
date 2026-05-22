"""Run SRF at VICE-matched dimensionalities (alpha=0).

For each data percentage, reads the median VICE dimensionality from
the VICE model directories, then runs SRF at that rank on all partitions.

Uses alpha=0, train_90.txt for 100% (no leak).

Outputs:
    outputs/vice_matched_alpha0/
        results.csv
        embeddings/*.npz
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils.helpers import compute_similarity_matrix_from_triplets

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PARTITION_DIR = PROJECT_ROOT / "data" / "things" / "partitions"
DATA_DIR = PROJECT_ROOT / "data" / "things"
BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
VICE_MODEL_DIR = Path(__file__).resolve().parent.parent / "vice" / "outputs" / "models"
LOG_PATH = BASE_OUTPUT_DIR / "vice_matched.log"

PARTITIONS = {5: 20, 10: 10, 20: 5, 50: 2, 100: 1}
N_OBJECTS = 1854
SEEDS = list(range(10))
ALPHA = 0.0

BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)


def load_partition_triplets(pct: int, part: int = 0):
    if pct == 100:
        triplet_dir = DATA_DIR / "triplets_47"
        train = np.loadtxt(triplet_dir / "train_90.txt", dtype=float).astype(int)
        val = np.loadtxt(triplet_dir / "test_10.txt", dtype=float).astype(int)
    else:
        d = PARTITION_DIR / f"{pct}pct_part{part}"
        train = np.loadtxt(d / "train_90.txt", dtype=float).astype(int)
        val = np.loadtxt(d / "test_10.txt", dtype=float).astype(int)
    return train, val


def compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    idx = triplets.astype(int)
    ei, ej, ek = embedding[idx[:, 0]], embedding[idx[:, 1]], embedding[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def get_vice_dims() -> dict[int, int]:
    """Read median VICE dimensionality per percentage from model directories."""
    dims = {}
    for pct in PARTITIONS:
        pct_dims = []
        for d in VICE_MODEL_DIR.iterdir():
            m = re.match(rf"vice_{pct}pct_part\d+_seed\d+", d.name)
            if not m:
                continue
            pf = list(d.glob("**/parameters.npz"))
            if not pf:
                continue
            emb = np.load(pf[0], allow_pickle=True).get("pruned_q_mu")
            if emb is not None:
                pct_dims.append(emb.shape[1])
        if pct_dims:
            dims[pct] = int(np.median(pct_dims))
    return dims


def run_single_srf(rsm, val, pct, part, seed, rank):
    model = SRF(rank=rank, random_state=seed, max_outer=200, max_inner=50, tol=1e-5)
    embedding = model.fit_transform(rsm)
    acc = compute_triplet_accuracy(embedding, val)
    return {
        "model": "SRF",
        "pct": pct,
        "part": part,
        "seed": seed,
        "rank": rank,
        "val_acc": acc,
        "n_iter": model.n_iter_,
    }, embedding


def main():
    output_dir = BASE_OUTPUT_DIR / "vice_matched_alpha0"
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)

    vice_dims = get_vice_dims()
    log.info(f"VICE dims: {vice_dims}")

    # Precompute RSMs
    log.info("Precomputing RSMs (alpha=0)...")
    rsm_cache = {}
    val_cache = {}
    for pct, n_parts in sorted(PARTITIONS.items()):
        for part in range(n_parts):
            train, val = load_partition_triplets(pct, part)
            rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train, alpha=ALPHA)
            rsm_cache[(pct, part)] = rsm
            val_cache[(pct, part)] = val

    # Build tasks
    tasks = []
    task_meta = []
    for pct, n_parts in sorted(PARTITIONS.items()):
        rank = vice_dims[pct]
        for part in range(n_parts):
            for seed in SEEDS:
                tasks.append((rsm_cache[(pct, part)], val_cache[(pct, part)],
                              pct, part, seed, rank))
                task_meta.append((pct, part, seed))

    log.info(f"Running {len(tasks)} SRF fits (alpha={ALPHA}, VICE-matched dims)...")
    raw_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_srf)(*t) for t in tasks
    )

    # Save
    records = []
    for (record, embedding), (pct, part, seed) in zip(raw_results, task_meta):
        records.append(record)
        np.savez_compressed(
            emb_dir / f"srf_{pct}pct_part{part}_seed{seed}.npz",
            embedding=embedding,
        )

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "results.csv", index=False)

    (output_dir / "vice_dims.json").write_text(json.dumps(vice_dims, indent=2))

    log.info("\nResults:")
    for pct in sorted(PARTITIONS):
        sub = df[df["pct"] == pct]
        log.info(f"  {pct}%: {sub['val_acc'].mean()*100:.2f}% +/- {sub['val_acc'].std()*100:.2f}% "
                 f"(k={sub['rank'].iloc[0]}, n={len(sub)})")
    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
