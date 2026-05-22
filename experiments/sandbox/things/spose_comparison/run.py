"""Compare SRF k=49 vs SPoSE 49d on 1.47M triplets.

Two RSM variants:
  1. RSM from full trainset (1.31M triplets)
  2. RSM from train_90 (90% of trainset, ~1.18M triplets)

Both evaluated on validationset.txt (146K truly held-out).
SPoSE trained on 90/10 split, so train_90 is the fair comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "things"
TRIPLET_DIR = DATA_DIR / "triplets_147"

RANK = 49
SEEDS = list(range(5))


def acc(emb, trips):
    idx = trips.astype(int)
    ei, ej, ek = emb[idx[:, 0]], emb[idx[:, 1]], emb[idx[:, 2]]
    sij = (ei * ej).sum(1)
    sik = (ei * ek).sum(1)
    sjk = (ej * ek).sum(1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_and_eval(rsm, val, seed, condition):
    model = SRF(rank=RANK, random_state=seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm)
    return {
        "model": "SRF",
        "condition": condition,
        "rank": RANK,
        "seed": seed,
        "val_acc": acc(w, val),
        "n_iter": model.n_iter_,
    }


def main():
    val = np.loadtxt(TRIPLET_DIR / "validationset.txt", dtype=int)
    train_full = np.loadtxt(TRIPLET_DIR / "trainset.txt", dtype=int)
    train_90 = np.loadtxt(TRIPLET_DIR / "train_90.txt", dtype=int)
    spose49 = np.loadtxt(DATA_DIR / "spose_embedding_49d.txt")

    log.info("val: %d, train_full: %d, train_90: %d", len(val), len(train_full), len(train_90))

    spose_val_acc = acc(spose49, val)
    log.info("SPoSE 49d on val: %.4f", spose_val_acc)

    log.info("Building RSMs...")
    rsm_full = compute_similarity_matrix_from_triplets(1854, train_full, alpha=0)
    rsm_90 = compute_similarity_matrix_from_triplets(1854, train_90, alpha=0)

    tasks = (
        [(rsm_full, val, s, "full_trainset") for s in SEEDS]
        + [(rsm_90, val, s, "train_90") for s in SEEDS]
    )
    log.info("Running %d SRF fits...", len(tasks))

    records = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_and_eval)(r, v, s, c) for r, v, s, c in tasks
    )

    records.append({
        "model": "SPoSE",
        "condition": "spose_49d",
        "rank": 49,
        "seed": 0,
        "val_acc": spose_val_acc,
        "n_iter": 0,
    })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    log.info("\nSPoSE 49d: %.4f", spose_val_acc)
    for cond in ["full_trainset", "train_90"]:
        sub = df[df["condition"] == cond]
        log.info("%s: %.4f +/- %.4f", cond, sub["val_acc"].mean(), sub["val_acc"].std())


if __name__ == "__main__":
    main()
