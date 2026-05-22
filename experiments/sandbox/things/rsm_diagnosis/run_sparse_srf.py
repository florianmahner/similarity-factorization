"""Test SRF with L1 sparsity penalty on THINGS triplets.

Uses the _bsum_sparse Cython module with l1_penalty integrated
into the BSUM quartic solver (d_coef += l1_penalty).
"""
import json
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf._bsum_sparse import update_w_blas_blocked as update_w_sparse
from pysrf import SRF

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
KAPPA_RANKS = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "srf" / "outputs" / "kappa_alpha0" / "ranks.json"
N = 1854


def acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_sparse_srf(s, rank, seed, l1, max_outer=500, max_inner=30, tol=1e-4):
    """Fit SRF with L1 penalty using the sparse BSUM solver."""
    from pysrf.model import _initialize_w, _frobenius_residual
    s = np.array(s, dtype=np.float64, order='C', copy=True)
    w = _initialize_w(s, rank, "random_sqrt", seed)
    w = np.array(w, dtype=np.float64, order='C', copy=True)
    for it in range(max_outer):
        w = update_w_sparse(s, w, max_iter=max_inner, tol=tol, l1_penalty=l1)
        res_norm, xhat_norm = _frobenius_residual(s, w)
        n = s.shape[0]
        eps_pri = n * tol + 1e-4 * max(np.linalg.norm(s, "fro"), xhat_norm)
        if res_norm <= eps_pri:
            break
    return w


def _fit_eval(s, rank, seed, val, l1, name):
    w = fit_sparse_srf(s, rank, seed, l1)
    sparsity = (w < 1e-4).mean()
    return name, seed, acc(w, val), sparsity


def main():
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)
    rank = json.load(open(KAPPA_RANKS))["100"]
    log.info(f"Rank: {rank}")

    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1); np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1); np.add.at(shown, (b, a), 1)
    s = np.divide(counts, shown, out=np.full((N, N), 0.5), where=shown > 0)
    np.fill_diagonal(s, 1.0)

    # VICE baseline
    vice_dir = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "vice" / "outputs" / "models"
    vice_accs = []
    for seed in range(10):
        d = np.load(
            vice_dir / f"vice_100pct_part0_seed{seed}" / "variational" / "4.12mio" / "sslab" / "90" / "256" / "1.0" / str(seed) / "params" / "parameters.npz",
            allow_pickle=True,
        )
        w = np.maximum(d["pruned_q_mu"], 0)
        vice_accs.append(acc(w, val))
        if seed == 0:
            vice_sparsity = (w < 1e-4).mean()
    log.info(f"VICE: {np.mean(vice_accs):.4f} +/- {np.std(vice_accs):.4f} (sparsity={vice_sparsity:.3f})")

    # SRF baseline (no L1)
    w_base = fit_sparse_srf(s, rank, 42, l1=0.0)
    log.info(f"SRF (l1=0):   {acc(w_base, val):.4f} (sparsity={(w_base < 1e-4).mean():.3f})")

    # Sweep L1 values (3 seeds each, parallel)
    l1_values = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    conditions = [(s, rank, seed, val, l1, f"l1={l1}")
                  for l1 in l1_values for seed in range(3)]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_eval)(*c) for c in conditions
    )

    log.info("\nResults:")
    for l1 in l1_values:
        sub = [(a, sp) for n, sd, a, sp in results if n == f"l1={l1}"]
        accs_list = [a for a, sp in sub]
        sparsities = [sp for a, sp in sub]
        log.info(f"  l1={l1:<6}: {np.mean(accs_list):.4f} +/- {np.std(accs_list):.4f} "
                 f"(sparsity={np.mean(sparsities):.3f})")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
