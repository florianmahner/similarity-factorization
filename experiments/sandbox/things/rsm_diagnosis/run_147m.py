"""SRF on 1.47M triplet dataset: kappa rank selection + SRF + compare to SPoSE 49d."""
import logging
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.coherence import (
    _estimate_kappa_hat,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_147"
N = 1854


def triplet_acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def _fit_one(s, rank, seed, val):
    model = SRF(rank=rank, random_state=seed, max_outer=500, max_inner=30, tol=1e-4, verbose=0)
    w = model.fit_transform(s)
    return rank, seed, triplet_acc(w, val)


def main():
    log.info("Loading 1.47M triplets...")
    train = np.loadtxt(DATA_DIR / "trainset.txt").astype(int)
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    log.info(f"Train: {len(train):,}, Val: {len(val):,}")

    # Build count RSM
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)

    s = (counts + 1) / (shown + 2)
    s[shown == 0] = 0.5
    np.fill_diagonal(s, 1.0)

    n_missing = (shown == 0).sum() // 2
    log.info(f"Missing pairs: {n_missing:,}, mean shown: {shown[shown > 0].mean():.1f}")

    # Step 1: Kappa rank selection
    log.info("\nEstimating rank via kappa...")
    k_list = list(range(1, 80, 1))
    p_list = np.linspace(0.05, 0.5, 30)

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list=k_list,
        p_list=p_list,
        B=100,
        random_state=42,
        compute_null=True,
        B_null=50,
        alpha_tau=0.05,
        ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=os.cpu_count() - 1,
        show_progress=True,
        visualize=False,
    )

    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_arr = result["k_list"]
    p_arr = result["p"]

    kappa_hat, kappa_info = _estimate_kappa_hat(x_median, p_arr, hi_band_quantile=0.85)
    k_star, cp_info = kappa_changepoint(kappa_hat, k_arr)
    log.info(f"k* = {k_star} (kappa changepoint)")

    # Step 2: SPoSE baseline
    spose = np.maximum(np.loadtxt(PROJECT_ROOT / "data" / "things" / "spose_embedding_49d.txt"), 0)
    log.info(f"\nSPoSE 49d: val={triplet_acc(spose, val):.4f}")

    # Step 3: SRF at k* (10 seeds, parallel)
    log.info(f"\nFitting SRF at rank={k_star} (10 seeds, parallel)...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_one)(s, k_star, seed, val) for seed in range(10)
    )

    accs = [acc for _, _, acc in results]
    for seed, acc in enumerate(accs):
        log.info(f"  seed={seed}: val={acc:.4f}")
    log.info(f"  mean: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
