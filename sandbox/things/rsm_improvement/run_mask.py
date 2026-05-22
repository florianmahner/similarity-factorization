"""Test masking chance-level RSM entries as missing before SRF."""

import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_spose_embedding, load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N = 1854
RANK = 45


def acc(emb, idx):
    ei, ej, ek = emb[idx[:, 0]], emb[idx[:, 1]], emb[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def run_one(s_masked, val_idx, label):
    model = SRF(
        rank=RANK, random_state=0,
        max_outer=50, max_inner=200,
        tol=1e-4, verbose=0,
    )
    w = model.fit_transform(s_masked)
    return label, acc(w, val_idx)


def main():
    train, val = load_triplets(DATA_DIR)
    val_idx = val.astype(int)
    spose = load_spose_embedding(DATA_DIR, num_dims=66)
    log.info("SPoSE: %.4f", acc(spose, val_idx))

    s = compute_similarity_matrix_from_triplets(N, train, alpha=0.0)
    nan_mask = np.isnan(s)
    diag = np.eye(N, dtype=bool)

    configs = []

    # Raw baseline (no masking)
    configs.append((s.copy(), "raw"))

    # Mask entries near 1/3
    for thresh in [0.05, 0.10]:
        s_masked = s.copy()
        mask = (np.abs(s_masked - 1 / 3) < thresh) & ~diag & ~nan_mask
        pct = mask.sum() / (~nan_mask & ~diag).sum() * 100
        s_masked[mask] = np.nan
        log.info("thresh=%.2f: masking %.1f%% as NaN", thresh, pct)
        configs.append((s_masked, f"mask_{thresh:.2f}"))

    results = Parallel(n_jobs=len(configs), verbose=0)(
        delayed(run_one)(sf, val_idx, label) for sf, label in configs
    )
    for label, a in results:
        log.info("SRF k=%d %12s: %.4f", RANK, label, a)


if __name__ == "__main__":
    main()
