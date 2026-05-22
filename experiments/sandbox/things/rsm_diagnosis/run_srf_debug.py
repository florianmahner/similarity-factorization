"""Debug: test if NaN handling / rho affects SRF accuracy at rank=24."""
import logging
from pathlib import Path

import numpy as np
from pysrf import SRF

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
N = 1854
RANK = 24


def triplet_acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def main():
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)
    test = np.loadtxt(DATA_DIR / "test_10.txt").astype(int)

    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)

    n_missing = (shown == 0).sum() // 2
    log.info(f"Missing pairs: {n_missing}")

    for fill in [0.0, 0.5]:
        s = (counts + 1) / (shown + 2)
        s[shown == 0] = fill
        np.fill_diagonal(s, 1.0)

        model = SRF(rank=RANK, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0)
        w = model.fit_transform(s)
        log.info(f"fill={fill}: test={triplet_acc(w, test):.4f} (iters={model.n_iter_})")


if __name__ == "__main__":
    main()
