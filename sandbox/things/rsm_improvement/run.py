"""Diagnostic: test embedding post-processing on existing rank sweep results.

Tests whether row-normalizing SRF embeddings (or other transforms) improves
triplet prediction accuracy without re-fitting SRF.

Theory: SRF has no sparsity/norm regularization, so row norms are a free
noise channel. SPoSE uses L1 which naturally constrains norms. If normalizing
SRF embeddings helps, it confirms norms are noisy.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.utils import get_output_dir
from src.utils.io import load_spose_embedding, load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

DATA_DIR = Path("data/things")
EMB_DIR = Path(
    "experiments/analyses/things_behavior/rank_sweep/outputs/embeddings"
)

RANKS = [5, 10, 15, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 66]
SEEDS = list(range(10))
ALPHA = 0


def compute_triplet_accuracy(embedding, triplets):
    idx = triplets.astype(int)
    ei, ej, ek = embedding[idx[:, 0]], embedding[idx[:, 1]], embedding[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def row_normalize(w):
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return w / norms


def soft_threshold(w, lam):
    return np.maximum(w - lam, 0)


def main():
    _, val_triplets = load_triplets(DATA_DIR)
    spose = load_spose_embedding(DATA_DIR, num_dims=66)
    vice = np.maximum(np.loadtxt(DATA_DIR / "vice_embedding_66d.txt"), 0)

    log.info("Validation triplets: %d", len(val_triplets))
    log.info("SPoSE raw:        %.4f", compute_triplet_accuracy(spose, val_triplets))
    log.info("SPoSE normalized: %.4f", compute_triplet_accuracy(row_normalize(spose), val_triplets))
    log.info("VICE raw:         %.4f", compute_triplet_accuracy(vice, val_triplets))
    log.info("VICE normalized:  %.4f", compute_triplet_accuracy(row_normalize(vice), val_triplets))
    log.info("")

    records = []

    for rank in RANKS:
        for seed in SEEDS:
            path = EMB_DIR / f"srf_alpha{ALPHA}_k{rank}_s{seed}.npz"
            if not path.exists():
                continue
            w = np.load(path)["embedding"]

            acc_raw = compute_triplet_accuracy(w, val_triplets)
            acc_norm = compute_triplet_accuracy(row_normalize(w), val_triplets)

            for lam in [0.001, 0.005, 0.01, 0.02, 0.05]:
                w_sparse = soft_threshold(w, lam)
                acc_sparse = compute_triplet_accuracy(w_sparse, val_triplets)
                records.append({
                    "rank": rank, "seed": seed,
                    "method": f"threshold_{lam}",
                    "val_acc": acc_sparse,
                })

            records.append({"rank": rank, "seed": seed, "method": "raw", "val_acc": acc_raw})
            records.append({"rank": rank, "seed": seed, "method": "normalized", "val_acc": acc_norm})

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "postprocessing_results.csv", index=False)

    log.info("=== Results by method (mean over seeds) ===")
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        summary = sub.groupby("rank")["val_acc"].mean()
        log.info("\n%s:", method)
        for rank, acc in summary.items():
            log.info("  k=%3d: %.4f", rank, acc)


if __name__ == "__main__":
    main()
