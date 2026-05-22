"""Run SRF low-data experiment at coherence-estimated ranks (kappa + PCT).

Same experimental setup as srf_seeds.py but uses ranks from the
permutation coherence test (PCT) and kappa changepoint instead of VICE dims.

Precomputes RSMs per partition (avoiding redundant triplet-to-RSM conversion),
then parallelizes across (partition, seed, rank, method) tasks.

Usage:
    poetry run python experiments/things_behavior/lowdata/srf_coherence_ranks.py --run
    poetry run python experiments/things_behavior/lowdata/srf_coherence_ranks.py --show
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

from ...ranks import load_things_ranks

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PARTITION_DIR = PROJECT_ROOT / "data" / "things" / "partitions"
DATA_DIR = PROJECT_ROOT / "data" / "things"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

PARTITIONS = {5: 20, 10: 10, 20: 5, 50: 2, 100: 1}
N_OBJECTS = 1854
SEEDS = list(range(10))


def load_partition_triplets(pct, part=0):
    if pct == 100:
        triplet_dir = DATA_DIR / "triplets_47"
        train = np.loadtxt(triplet_dir / "train_90.txt", dtype=float).astype(int)
        val = np.loadtxt(triplet_dir / "test_10.txt", dtype=float).astype(int)
    else:
        d = PARTITION_DIR / f"{pct}pct_part{part}"
        train = np.loadtxt(d / "train_90.txt", dtype=float).astype(int)
        val = np.loadtxt(d / "test_10.txt", dtype=float).astype(int)
    return train, val


def compute_triplet_accuracy(embedding, triplets):
    idx = triplets.astype(int)
    ei = embedding[idx[:, 0]]
    ej = embedding[idx[:, 1]]
    ek = embedding[idx[:, 2]]
    sim_ij = np.sum(ei * ej, axis=1)
    sim_ik = np.sum(ei * ek, axis=1)
    sim_jk = np.sum(ej * ek, axis=1)
    return float(np.mean((sim_ij > sim_ik) & (sim_ij > sim_jk)))


def run_single(rsm, val_triplets, pct, part, seed, rank, method):
    model = SRF(rank=rank, random_state=seed, max_outer=200, max_inner=50, tol=1e-5)
    embedding = model.fit_transform(rsm)
    val_acc = compute_triplet_accuracy(embedding, val_triplets)
    train_acc = 0.0  # skip train acc to save time
    return {
        "method": method, "model": "SRF",
        "pct": pct, "part": part, "seed": seed, "rank": rank,
        "val_acc": val_acc, "train_acc": train_acc,
        "n_iter": model.n_iter_,
        "n_val": len(val_triplets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    if args.show:
        kappa_ranks = load_things_ranks(method="kappa")
        total = sum(n * len(SEEDS) * 2 for n in PARTITIONS.values())
        print(f"Total tasks: {total}")
        for pct in sorted(PARTITIONS):
            n = PARTITIONS[pct] * len(SEEDS) * 2
            print(f"  {pct}%: kappa rank={kappa_ranks[pct]}, {n} tasks")
        return

    if not args.run:
        parser.print_help()
        return

    kappa_ranks = load_things_ranks(method="kappa")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.csv"

    # Precompute RSMs and load val triplets per (pct, part)
    # This is the slow step -- do it ONCE, not per worker
    print("Precomputing RSMs...")
    rsm_cache = {}
    val_cache = {}
    for pct, n_parts in sorted(PARTITIONS.items()):
        for part in range(n_parts):
            train, val = load_partition_triplets(pct, part)
            rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train, alpha=1.0)
            rsm_cache[(pct, part)] = rsm
            val_cache[(pct, part)] = val
            print(f"  {pct}% part {part}: RSM {rsm.shape}, val {len(val)} triplets")

    # Build tasks
    tasks = []
    for pct, n_parts in sorted(PARTITIONS.items()):
        for part in range(n_parts):
            rsm = rsm_cache[(pct, part)]
            val = val_cache[(pct, part)]
            for seed in SEEDS:
                tasks.append((rsm, val, pct, part, seed, kappa_ranks[pct], "kappa"))

    # Check for completed tasks
    completed = set()
    if output_path.exists():
        df_done = pd.read_csv(output_path)
        completed = {(r.pct, r.part, r.seed, r.rank, r.method) for r in df_done.itertuples()}
    pending = [t for t in tasks if (t[2], t[3], t[4], t[5], t[6]) not in completed]

    print(f"Total: {len(tasks)}, Completed: {len(completed)}, Pending: {len(pending)}")
    print(f"Kappa ranks: {kappa_ranks}")

    if not pending:
        print("All done!")
        df = pd.read_csv(output_path)
        print(df.groupby(["method", "pct", "rank"])["val_acc"].agg(["mean", "std", "count"]).round(4))
        return

    print(f"Running {len(pending)} tasks with n_jobs={args.n_jobs}...")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_single)(*t) for t in pending
    )

    df = pd.DataFrame(results)
    df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)

    print(f"\nResults saved to {output_path}")
    df_all = pd.read_csv(output_path)
    print(df_all.groupby(["method", "pct", "rank"])["val_acc"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    main()
