"""Test SRF convergence at 100% data with PCT rank=25.

Goal: check if stronger convergence can surpass VICE (65.03%) at full data.
"""

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things"
VAL_PATH = DATA_DIR / "partitions" / "50pct_part0" / "test_10.txt"

RANK = 25
VICE_ACC = 0.6503
SEEDS = list(range(20))


def triplet_accuracy(emb, trips):
    idx = trips.astype(int)
    ei, ej, ek = emb[idx[:, 0]], emb[idx[:, 1]], emb[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def run_single(rsm, val, seed, max_outer, max_inner, tol):
    model = SRF(
        rank=RANK, random_state=seed,
        max_outer=max_outer, max_inner=max_inner, tol=tol,
    )
    w = model.fit_transform(rsm)
    a = triplet_accuracy(w, val)
    return {
        "seed": seed,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "tol": tol,
        "rank": RANK,
        "val_acc": a,
        "n_iter": model.n_iter_,
    }


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info("Loading data...")
    triplets, _ = load_triplets(DATA_DIR)
    val = np.loadtxt(VAL_PATH, dtype=float).astype(int)
    rsm = compute_similarity_matrix_from_triplets(1854, triplets, alpha=1.0)
    log.info(f"RSM: {rsm.shape}, val: {len(val)} triplets")

    log.info(f"Running {len(SEEDS)} seeds at rank={RANK}, max_outer=5000, max_inner=200, tol=1e-7")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single)(rsm, val, seed, 5000, 200, 1e-7)
        for seed in SEEDS
    )

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    log.info(f"\nResults (VICE = {VICE_ACC*100:.2f}%):")
    for _, row in df.iterrows():
        marker = ">>>" if row["val_acc"] > VICE_ACC else "   "
        log.info(f"  {marker} seed={row['seed']:2d}: {row['val_acc']*100:.2f}% (n_iter={row['n_iter']})")

    log.info(f"\nMean: {df['val_acc'].mean()*100:.2f}% +/- {df['val_acc'].std()*100:.2f}%")
    log.info(f"Best: {df['val_acc'].max()*100:.2f}% (seed={df.loc[df['val_acc'].idxmax(), 'seed']})")
    log.info(f"VICE: {VICE_ACC*100:.2f}%")


if __name__ == "__main__":
    main()
