"""Test masking unreliable RSM entries for ADMM imputation."""
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


def acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def main():
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)

    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1); np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1); np.add.at(shown, (b, a), 1)

    s = np.divide(counts, shown, out=np.full((N, N), np.nan), where=shown > 0)
    np.fill_diagonal(s, 1.0)

    log.info("Masking low-shown entries (ADMM imputation, alpha=0)")
    for min_shown in [2, 3, 5, 7]:
        s_m = s.copy()
        unreliable = (shown > 0) & (shown < min_shown)
        unreliable = unreliable | unreliable.T
        s_m[unreliable] = np.nan
        n_nan = np.isnan(s_m).sum() // 2
        w = SRF(rank=24, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0).fit_transform(s_m)
        log.info(f"  min_shown={min_shown} ({n_nan:,} NaN): {acc(w, val):.4f}")

    n_nan_base = np.isnan(s).sum() // 2
    w_base = SRF(rank=24, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0).fit_transform(s)
    log.info(f"  baseline ({n_nan_base:,} NaN): {acc(w_base, val):.4f}")


if __name__ == "__main__":
    main()
