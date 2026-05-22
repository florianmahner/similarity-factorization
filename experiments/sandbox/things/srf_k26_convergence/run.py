"""Test if more iterations at k=26 improves THINGS triplet accuracy.

The archived kappa runs used max_outer=200 and ALL hit the limit.
Standard SRF uses max_outer=2000. Test whether convergence helps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things"
PARTITION_DIR = DATA_DIR / "partitions"

OUTPUT_DIR = get_output_dir()

RANK = 26
SEEDS = list(range(10))
MAX_OUTERS = [200, 500, 1000, 2000]


def _load_data():
    train, _ = load_triplets(DATA_DIR)
    val = np.loadtxt(PARTITION_DIR / "50pct_part0" / "test_10.txt",
                     dtype=float).astype(int)
    rsm = compute_similarity_matrix_from_triplets(1854, train, alpha=1.0)
    return rsm, val


def _triplet_accuracy(embedding, triplets):
    idx = triplets.astype(int)
    ei, ej, ek = embedding[idx[:, 0]], embedding[idx[:, 1]], embedding[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def _run_one(rsm, val, seed, max_outer):
    model = SRF(rank=RANK, random_state=seed, max_outer=max_outer,
                max_inner=50, tol=1e-5, verbose=0)
    w = model.fit_transform(rsm)
    acc = _triplet_accuracy(w, val)
    return {"seed": seed, "max_outer": max_outer, "n_iter": model.n_iter_,
            "val_acc": acc}


def main():
    print("Loading data...")
    rsm, val = _load_data()
    print(f"RSM: {rsm.shape}, val triplets: {len(val)}")

    conditions = [(seed, mo) for mo in MAX_OUTERS for seed in SEEDS]
    print(f"Running {len(conditions)} conditions (k={RANK})...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_one)(rsm, val, seed, mo) for seed, mo in conditions
    )

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    print("\n=== Results ===")
    print(df.groupby("max_outer")[["val_acc", "n_iter"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
