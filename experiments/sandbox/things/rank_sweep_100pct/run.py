"""Rank sweep: SRF accuracy vs rank on 100% THINGS data.

Does increasing rank beyond kappa k*=26 improve triplet prediction?
If SRF at k=30-50 surpasses VICE (0.6503), then kappa is too conservative
and we should explore pushing rank selection higher. If not, the 26-dim
result is not a limitation.

Ranks: 20, 25, 26, 28, 30, 33, 36, 40, 45, 50, 55, 60, 66, 80
Seeds: 10 per rank (enough for stable mean + CI)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things"
VAL_PATH = DATA_DIR / "partitions" / "50pct_part0" / "test_10.txt"

RANKS = [20, 25, 26, 28, 30, 33, 36, 40, 45, 50, 60, 66]
SEEDS = list(range(5))
N_OBJECTS = 1854

VICE_ACC = 0.6503
SPOSE_ACC = 0.6437


def triplet_accuracy(emb, trips):
    idx = trips.astype(int)
    ei, ej, ek = emb[idx[:, 0]], emb[idx[:, 1]], emb[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def run_single(rsm, val, rank, seed):
    model = SRF(rank=rank, random_state=seed, max_outer=200, max_inner=50, tol=1e-5)
    w = model.fit_transform(rsm)
    acc = triplet_accuracy(w, val)
    return {
        "rank": rank,
        "seed": seed,
        "val_acc": acc,
        "n_iter": model.n_iter_,
    }


def main():
    log.info("Loading data...")
    triplets, _ = load_triplets(DATA_DIR)
    val = np.loadtxt(VAL_PATH, dtype=float).astype(int)
    rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, triplets, alpha=1.0)
    log.info(f"RSM: {rsm.shape}, val: {len(val)} triplets")

    tasks = [(rsm, val, rank, seed) for rank in RANKS for seed in SEEDS]
    log.info(f"Running {len(tasks)} tasks ({len(RANKS)} ranks x {len(SEEDS)} seeds)")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single)(r, v, rank, seed) for r, v, rank, seed in tasks
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    summary = df.groupby("rank")["val_acc"].agg(["mean", "std", "count"])
    summary.to_csv(OUTPUT_DIR / "summary.csv")

    log.info(f"\nRank sweep results (VICE={VICE_ACC*100:.2f}%, SPoSE={SPOSE_ACC*100:.2f}%):")
    log.info(f"{'Rank':>6s} {'Mean':>8s} {'Std':>8s} {'vs VICE':>10s}")
    for rank, row in summary.iterrows():
        delta = (row["mean"] - VICE_ACC) * 100
        marker = ">>>" if row["mean"] > VICE_ACC else "   "
        log.info(
            f"  {marker} {rank:>4d}  {row['mean']*100:>7.2f}%  {row['std']*100:>7.3f}%  {delta:>+7.2f} pp"
        )


if __name__ == "__main__":
    main()
