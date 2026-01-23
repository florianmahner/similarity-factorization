"""Run SRF low-data experiment with multiple seeds per partition.

Matches VICE experimental setup: 10 seeds per partition.

Usage:
    # Run all seeds
    poetry run python experiments/things_behavior/srf_lowdata_seeds.py --run

    # Show distribution
    poetry run python experiments/things_behavior/srf_lowdata_seeds.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils.helpers import compute_similarity_matrix_from_triplets

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARTITION_DIR = PROJECT_ROOT / "data" / "things" / "partitions"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "things_behavior" / "srf_lowdata"

PARTITIONS = {
    5: 20,
    10: 10,
    20: 5,
    50: 2,
}

N_OBJECTS = 1854
SEEDS = list(range(10))  # Seeds 0-9

# VICE's estimated dimensionality per fraction (mean across partitions)
VICE_DIMS = {5: 11, 10: 20, 20: 41, 50: 69}


def load_partition_triplets(pct: int, part: int) -> tuple[np.ndarray, np.ndarray]:
    """Load train/val triplets for a partition."""
    partition_dir = PARTITION_DIR / f"{pct}pct_part{part}"
    train = np.loadtxt(partition_dir / "train_90.txt", dtype=float).astype(int)
    val = np.loadtxt(partition_dir / "test_10.txt", dtype=float).astype(int)
    return train, val


def compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    """Evaluate odd-one-out accuracy using softmax over dot product similarities."""
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


def run_single(pct: int, part: int, seed: int) -> dict:
    """Run SRF on a single partition with specific seed."""
    rank = VICE_DIMS[pct]
    train_triplets, val_triplets = load_partition_triplets(pct, part)
    rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train_triplets, alpha=1.0)

    model = SRF(rank=rank, random_state=seed, max_outer=100, max_inner=30)
    embedding = model.fit_transform(rsm)

    val_acc = compute_triplet_accuracy(embedding, val_triplets)
    train_acc = compute_triplet_accuracy(embedding, train_triplets)

    return {
        "model": "SRF",
        "pct": pct,
        "part": part,
        "seed": seed,
        "rank": rank,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "n_train": len(train_triplets),
        "n_val": len(val_triplets),
    }


def get_all_tasks() -> list[tuple[int, int, int]]:
    """Get all (pct, part, seed) tasks."""
    tasks = []
    for pct, n_parts in PARTITIONS.items():
        for part in range(n_parts):
            for seed in SEEDS:
                tasks.append((pct, part, seed))
    return tasks


def get_completed_tasks(output_path: Path) -> set[tuple[int, int, int]]:
    """Load already completed tasks from CSV."""
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    return {(row.pct, row.part, row.seed) for row in df.itertuples()}


def run_single_and_save(pct: int, part: int, seed: int, output_path: Path) -> dict:
    """Run single task and append result to CSV."""
    result = run_single(pct, part, seed)
    df = pd.DataFrame([result])
    df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run all tasks")
    parser.add_argument("--show", action="store_true", help="Show task count")
    parser.add_argument("--n-jobs", type=int, default=50, help="Number of parallel jobs")
    args = parser.parse_args()

    if args.show:
        tasks = get_all_tasks()
        print(f"Total tasks: {len(tasks)}")
        for pct, n_parts in PARTITIONS.items():
            print(f"  {pct}%: {n_parts} partitions × {len(SEEDS)} seeds = {n_parts * len(SEEDS)} tasks")
        return

    if args.run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "srf_lowdata_seeds.csv"

        all_tasks = get_all_tasks()
        completed = get_completed_tasks(output_path)
        pending = [t for t in all_tasks if t not in completed]

        print(f"Total: {len(all_tasks)}, Completed: {len(completed)}, Pending: {len(pending)}")

        if not pending:
            print("All tasks already completed!")
            df = pd.read_csv(output_path)
            print("\nSummary:")
            print(df.groupby(["pct"])["val_acc"].agg(["mean", "std", "count"]))
            return

        print(f"Running {len(pending)} SRF tasks with {args.n_jobs} parallel jobs...")

        # Run in batches to save intermediate results
        batch_size = args.n_jobs
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            results = Parallel(n_jobs=args.n_jobs, verbose=1)(
                delayed(run_single)(pct, part, seed) for pct, part, seed in batch
            )
            df = pd.DataFrame(results)
            df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
            print(f"Saved batch {batch_start // batch_size + 1}, total: {batch_start + len(batch)}/{len(pending)}")

        print(f"\nResults saved to {output_path}")
        df = pd.read_csv(output_path)
        print("\nSummary:")
        print(df.groupby(["pct"])["val_acc"].agg(["mean", "std", "count"]))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
