"""SRF lowdata: compare alpha=0 vs alpha=1 with kappa-estimated ranks.

For each alpha in {0.0, 1.0}:
  1. Estimate kappa rank per data percentage (sequential, each uses all cores)
  2. Run SRF at kappa ranks on all partitions x seeds (one big parallel batch)
  3. Save results.csv, ranks.json, embeddings/*.npz

Outputs:
  outputs/kappa_alpha0/  {ranks.json, results.csv, embeddings/*.npz}
  outputs/kappa_alpha1/  {ranks.json, results.csv, embeddings/*.npz}

Usage:
    ./scripts/submit experiments/analyses/things_behavior/lowdata/srf/run_alpha_comparison.py --bg
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    _estimate_kappa_hat,
    kappa_changepoint,
)
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PARTITION_DIR = PROJECT_ROOT / "data" / "things" / "partitions"
DATA_DIR = PROJECT_ROOT / "data" / "things"
BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LOG_PATH = BASE_OUTPUT_DIR / "alpha_comparison.log"

PARTITIONS = {5: 20, 10: 10, 20: 5, 50: 2, 100: 1}
N_OBJECTS = 1854
SEEDS = list(range(10))
ALPHAS = [0.0]

# Set up file logging so we can see what happened
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


def estimate_kappa_rank(rsm: np.ndarray) -> int:
    k_list = list(range(1, 61))
    p_list = np.linspace(0.05, 0.95, 25)
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        rsm, k_list=k_list, p_list=p_list,
        B=30, random_state=42,
        compute_null=True, B_null=20,
        alpha_tau=0.95, ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=140, show_progress=True, visualize=False,
    )
    diag = result["diagnostics"]
    kappa_hat, _ = _estimate_kappa_hat(diag["x_median"], result["p"], hi_band_quantile=0.85)
    k_star, _ = kappa_changepoint(kappa_hat, result["k_list"])
    return int(k_star)


def run_single_srf(rsm, val, alpha, pct, part, seed, rank):
    model = SRF(rank=rank, random_state=seed, max_outer=200, max_inner=50, tol=1e-5)
    embedding = model.fit_transform(rsm)
    acc = compute_triplet_accuracy(embedding, val)
    return {
        "alpha": alpha, "model": "SRF",
        "pct": pct, "part": part, "seed": seed, "rank": rank,
        "val_acc": acc, "n_iter": model.n_iter_, "n_val": len(val),
    }, embedding


def main():
    log.info("=" * 60)
    log.info("ALPHA COMPARISON: kappa ranks + SRF lowdata")
    log.info("=" * 60)

    all_ranks = {}

    # --- Phase 1: Kappa ranks (sequential over alpha x pct, each uses all cores) ---
    for alpha in ALPHAS:
        all_ranks[alpha] = {}
        for pct in sorted(PARTITIONS.keys()):
            log.info(f"Kappa: alpha={alpha}, {pct}%...")
            train, _ = load_partition_triplets(pct, part=0)
            rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train, alpha=alpha)
            k_star = estimate_kappa_rank(rsm)
            all_ranks[alpha][pct] = k_star
            log.info(f"  -> k* = {k_star}")

        # Save ranks immediately
        tag = f"alpha{int(alpha)}"
        out = BASE_OUTPUT_DIR / f"kappa_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "embeddings").mkdir(exist_ok=True)
        (out / "ranks.json").write_text(json.dumps(all_ranks[alpha], indent=2))
        log.info(f"Saved ranks for alpha={alpha}: {all_ranks[alpha]}")

    # --- Phase 2: Precompute all RSMs ---
    log.info("Precomputing RSMs...")
    rsm_cache = {}
    val_cache = {}
    for alpha in ALPHAS:
        for pct, n_parts in sorted(PARTITIONS.items()):
            for part in range(n_parts):
                train, val = load_partition_triplets(pct, part)
                rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train, alpha=alpha)
                rsm_cache[(alpha, pct, part)] = rsm
                val_cache[(pct, part)] = val
    log.info(f"Cached {len(rsm_cache)} RSMs")

    # --- Phase 3: All SRF fits in one batch ---
    tasks = []
    task_meta = []
    for alpha in ALPHAS:
        for pct, n_parts in sorted(PARTITIONS.items()):
            rank = all_ranks[alpha][pct]
            for part in range(n_parts):
                for seed in SEEDS:
                    tasks.append((
                        rsm_cache[(alpha, pct, part)],
                        val_cache[(pct, part)],
                        alpha, pct, part, seed, rank,
                    ))
                    task_meta.append((alpha, pct, part, seed))

    log.info(f"Running {len(tasks)} SRF fits...")
    raw_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_srf)(*t) for t in tasks
    )

    # --- Phase 4: Save per alpha ---
    log.info("Saving results and embeddings...")
    for alpha in ALPHAS:
        tag = f"alpha{int(alpha)}"
        out = BASE_OUTPUT_DIR / f"kappa_{tag}"
        emb_dir = out / "embeddings"

        records = []
        for (record, embedding), (a, pct, part, seed) in zip(raw_results, task_meta):
            if a != alpha:
                continue
            records.append(record)
            np.savez_compressed(
                emb_dir / f"srf_{pct}pct_part{part}_seed{seed}.npz",
                embedding=embedding,
            )

        df = pd.DataFrame(records)
        df.to_csv(out / "results.csv", index=False)
        summary = df.groupby(["pct", "rank"])["val_acc"].agg(["mean", "std", "count"])
        log.info(f"\n=== Alpha={alpha} ===\nRanks: {all_ranks[alpha]}\n{summary.round(4)}")

    log.info("Done!")


if __name__ == "__main__":
    main()
