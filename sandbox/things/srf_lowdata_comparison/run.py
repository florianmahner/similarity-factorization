"""Compare SRF vs VICE on low-data THINGS triplet task.

For each training percentage and partition:
1. Build RSM from training triplets
2. Run SRF with rank = VICE's estimated dimensionality
3. Evaluate odd-one-out accuracy on validation triplets
4. Compare with VICE results
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.colors import TEAL, ROSE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils.helpers import compute_similarity_matrix_from_triplets

OUTPUT_DIR = get_output_dir()

DATA_DIR = Path("/LOCAL/fmahner/similarity-factorization/data/things/partitions")
N_OBJECTS = 1854

VICE_DIMS = {5: 11, 10: 20, 20: 41, 50: 69}


def load_partition_triplets(pct: int, part: int) -> tuple[np.ndarray, np.ndarray]:
    """Load train/val triplets for a partition."""
    partition_dir = DATA_DIR / f"{pct}pct_part{part}"
    train = np.loadtxt(partition_dir / "train_90.txt", dtype=int)
    val = np.loadtxt(partition_dir / "test_10.txt", dtype=int)
    return train, val


def compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    """Evaluate odd-one-out accuracy from embedding using softmax.

    Triplet format: [i, j, k] where (i,j) is chosen pair, k is odd-one-out.
    Uses dot product similarity with softmax, predicting highest similarity pair.
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


def run_single(pct: int, part: int, rank: int) -> dict:
    """Run SRF on a single partition."""
    train_triplets, val_triplets = load_partition_triplets(pct, part)

    rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train_triplets, alpha=1.0)

    model = SRF(rank=rank, random_state=42, max_outer=100, max_inner=30)
    embedding = model.fit_transform(rsm)

    val_acc = compute_triplet_accuracy(embedding, val_triplets)
    train_acc = compute_triplet_accuracy(embedding, train_triplets)

    return {
        "pct": pct,
        "part": part,
        "rank": rank,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "n_train": len(train_triplets),
        "n_val": len(val_triplets),
    }


def get_partitions() -> list[tuple[int, int]]:
    """Get all (pct, part) combinations from the data directory."""
    partitions = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and "pct_part" in d.name:
            parts = d.name.split("pct_part")
            pct = int(parts[0])
            part = int(parts[1])
            partitions.append((pct, part))
    return sorted(partitions)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    partitions = get_partitions()
    print(f"Found {len(partitions)} partitions")

    tasks = [(pct, part, VICE_DIMS[pct]) for pct, part in partitions]

    results = Parallel(n_jobs=16, verbose=10)(
        delayed(run_single)(pct, part, rank) for pct, part, rank in tasks
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "srf_lowdata_results.csv", index=False)

    print("\nSRF Results:")
    print(df.groupby("pct")[["val_acc", "train_acc"]].agg(["mean", "std"]))

    vice_df = pd.read_csv(
        "/LOCAL/fmahner/similarity-factorization/sandbox/things/vice_lowdata_analysis/outputs/260120_140901/vice_lowdata_results.csv"
    )

    srf_agg = df.groupby("pct").agg({
        "val_acc": ["mean", "std"],
    }).reset_index()
    srf_agg.columns = ["pct", "srf_val_mean", "srf_val_std"]

    vice_agg = vice_df.groupby("pct").agg({
        "val_acc": ["mean", "std"],
    }).reset_index()
    vice_agg.columns = ["pct", "vice_val_mean", "vice_val_std"]

    merged = srf_agg.merge(vice_agg, on="pct")
    merged.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    print("\nComparison:")
    print(merged)

    fig, ax = create_figure("single")

    x = merged["pct"]
    width = 2

    ax.errorbar(
        x - width/2, merged["srf_val_mean"], yerr=merged["srf_val_std"],
        fmt="o-", color=TEAL, capsize=4, markersize=8, linewidth=2, label="SRF"
    )
    ax.errorbar(
        x + width/2, merged["vice_val_mean"], yerr=merged["vice_val_std"],
        fmt="s-", color=ROSE, capsize=4, markersize=8, linewidth=2, label="VICE"
    )

    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Validation accuracy (odd-one-out)")
    ax.set_xticks(x)
    ax.legend()
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "srf_vs_vice_accuracy.png")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
