"""Test principled RSM denoising to close the SRF-VICE gap.

Baseline: SRF on raw RSM (alpha=1, k=40) -> 63.98%
Target: VICE -> 64.15%

All RSMs precomputed first, then all SRF fits in one parallel batch.
"""

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

from src.utils import get_output_dir

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things"
N = 1854


def load_data():
    train = np.loadtxt(DATA_DIR / "triplets_47" / "train_90.txt", dtype=float).astype(int)
    val = np.loadtxt(DATA_DIR / "triplets_47" / "test_10.txt", dtype=float).astype(int)
    return train, val


def build_rsm(train, alpha=1.0):
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)
    rsm = np.divide(counts + alpha, shown + 2 * alpha, out=np.zeros((N, N)), where=shown > 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def triplet_acc(emb, trips):
    ei, ej, ek = emb[trips[:, 0]], emb[trips[:, 1]], emb[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def gavish_donoho_denoise(rsm):
    """Optimal eigenvalue shrinkage (Gavish & Donoho 2014, Frobenius loss)."""
    diag_orig = np.diag(rsm).copy()
    evals, evecs = np.linalg.eigh(rsm)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    n = len(evals)

    bulk = evals[int(0.2 * n):]
    med = np.median(bulk)
    mad = np.median(np.abs(bulk - med))
    sigma = mad / 0.6745

    threshold = 2 * np.sqrt(n) * sigma
    n_signal = int(np.sum(evals > threshold))

    evals_shrunk = np.zeros_like(evals)
    for i in range(n_signal):
        evals_shrunk[i] = np.sqrt(max(evals[i] ** 2 - 4 * n * sigma ** 2, 0))

    rsm_clean = (evecs * evals_shrunk) @ evecs.T
    rsm_clean = np.maximum(rsm_clean, 0)
    np.fill_diagonal(rsm_clean, diag_orig)
    log.info(f"  GD: sigma={sigma:.4f}, threshold={threshold:.2f}, kept={n_signal}")
    return rsm_clean


def run_one(rsm, val, rank, label):
    m = SRF(rank=rank, random_state=42, max_outer=100, max_inner=30, tol=1e-4)
    w = m.fit_transform(rsm)
    return {"label": label, "rank": rank, "val_acc": triplet_acc(w, val)}


def main():
    train, val = load_data()
    log.info(f"Train: {len(train)}, Val: {len(val)}")

    # Phase 1: Build all RSMs (fast, sequential)
    log.info("Building RSMs...")
    rsms = {}
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        rsms[f"alpha={alpha}"] = build_rsm(train, alpha=alpha)

    rsm_raw = rsms["alpha=1.0"]
    rsm_gd = gavish_donoho_denoise(rsm_raw)
    rsms["GD(alpha=1)"] = rsm_gd

    for alpha in [0.0, 0.5, 2.0]:
        rsm_a = build_rsm(train, alpha=alpha)
        rsms[f"GD(alpha={alpha})"] = gavish_donoho_denoise(rsm_a)

    # Phase 2: All SRF fits in one parallel batch
    tasks = []
    for label, rsm in rsms.items():
        for rank in [40]:
            tasks.append((rsm, val, rank, label))
    # Also test GD at higher ranks
    for rank in [50, 66]:
        tasks.append((rsms["GD(alpha=1)"], val, rank, f"GD(alpha=1) k={rank}"))

    log.info(f"Running {len(tasks)} SRF fits in parallel...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one)(r, v, k, l) for r, v, k, l in tasks
    )

    log.info(f"\nResults (all at k=40 unless noted, VICE=64.15%):")
    for r in sorted(results, key=lambda x: -x["val_acc"]):
        marker = ">>>" if r["val_acc"] > 0.6415 else "   "
        log.info(f"  {marker} {r['label']:>20s} k={r['rank']:>2d}: {r['val_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
