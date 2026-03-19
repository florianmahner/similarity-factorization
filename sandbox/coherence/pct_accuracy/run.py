"""Quick single-run accuracy at coherence-estimated ranks vs existing ranks."""

import logging
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()
PARTITION_DIR = Path("data/things/partitions")
N = 1854

CONFIGS = [
    (5,   [4, 6, 10]),
    (20,  [14, 11, 31]),
    (50,  [20, 21, 53]),
    (100, [25, 26, 66]),
]

LABELS = {4: "PCT", 14: "PCT", 20: "PCT", 25: "PCT",
          6: "kappa", 11: "kappa", 21: "kappa", 26: "kappa",
          10: "SRF-existing", 31: "SRF-existing", 53: "SRF-existing", 66: "SRF-existing"}


def load_triplets_pct(pct, part=0):
    if pct == 100:
        train, _ = load_triplets(Path("data/things"))
        val = np.loadtxt(PARTITION_DIR / "50pct_part0" / "test_10.txt", dtype=float).astype(int)
    else:
        d = PARTITION_DIR / f"{pct}pct_part{part}"
        train = np.loadtxt(d / "train_90.txt", dtype=float).astype(int)
        val = np.loadtxt(d / "test_10.txt", dtype=float).astype(int)
    return train, val


def triplet_acc(emb, triplets):
    n_correct = 0
    for i, j, k in triplets:
        sims = np.array([emb[i] @ emb[j], emb[i] @ emb[k], emb[j] @ emb[k]])
        p = np.exp(sims - sims.max())
        p /= p.sum()
        if np.argmax(p) == 0:
            n_correct += 1
    return n_correct / len(triplets)


def run_one(pct, rank, train, val):
    rsm = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    model = SRF(rank=rank, random_state=42, max_outer=100, max_inner=30)
    emb = model.fit_transform(rsm)
    acc = triplet_acc(emb, val)
    return {"pct": pct, "rank": rank, "val_acc": acc, "label": LABELS.get(rank, "")}


def main():
    # Preload all data
    data = {}
    for pct, ranks in CONFIGS:
        train, val = load_triplets_pct(pct)
        data[pct] = (train, val)
        log.info(f"Loaded {pct}% triplets")

    # Build all tasks
    tasks = []
    for pct, ranks in CONFIGS:
        train, val = data[pct]
        for rank in ranks:
            tasks.append((pct, rank, train, val))

    log.info(f"Running {len(tasks)} tasks in parallel")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one)(pct, rank, train, val)
        for pct, rank, train, val in tasks
    )

    log.info(f"\n{'pct':>4s}  {'rank':>5s}  {'val_acc':>8s}  {'label':>15s}")
    log.info("-" * 40)
    for r in sorted(results, key=lambda x: (x["pct"], x["rank"])):
        log.info(f"{r['pct']:>4d}  {r['rank']:>5d}  {r['val_acc']:>8.4f}  {r['label']:>15s}")


if __name__ == "__main__":
    main()
