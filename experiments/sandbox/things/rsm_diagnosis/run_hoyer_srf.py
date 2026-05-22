"""SRF with Hoyer sparseness projection after each BSUM step.

Hoyer sparseness: sp(x) = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)
After each BSUM iteration, project each row of W onto the set
{z >= 0 : ||z||_2 = ||w_i||_2, sp(z) = target}.
This preserves energy while concentrating it into fewer dimensions.
"""
import json
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf._bsum import update_w_blas_blocked as bsum_update
from pysrf.model import _initialize_w, _frobenius_residual

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


def hoyer_sparseness(x):
    """Hoyer sparseness measure in [0, 1]."""
    n = len(x)
    l1 = np.sum(np.abs(x))
    l2 = np.sqrt(np.sum(x ** 2))
    if l2 == 0:
        return 0.0
    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)


def project_w(w, target_sparseness):
    """Vectorized Hoyer projection: all rows at once."""
    n_rows, k = w.shape
    l2 = np.sqrt(np.sum(w ** 2, axis=1, keepdims=True))  # (n, 1)
    l2 = np.maximum(l2, 1e-12)
    target_l1 = l2 * (np.sqrt(k) - target_sparseness * (np.sqrt(k) - 1))  # (n, 1)

    z = w.copy()
    z += (target_l1 - z.sum(axis=1, keepdims=True)) / k

    for _ in range(50):
        neg = z < 0
        if not neg.any():
            break
        z[neg] = 0
        n_active = (~neg).sum(axis=1, keepdims=True).astype(float)
        n_active = np.maximum(n_active, 1)
        deficit = target_l1 - z.sum(axis=1, keepdims=True)
        adjustment = deficit / n_active
        z += adjustment * (~neg)

    cur_l2 = np.sqrt(np.sum(z ** 2, axis=1, keepdims=True))
    cur_l2 = np.maximum(cur_l2, 1e-12)
    z *= l2 / cur_l2
    return z


def fit_hoyer_srf(s, rank, seed, target_sp, max_outer=500, max_inner=30, tol=1e-4):
    """SRF with Hoyer projection after each BSUM outer iteration."""
    s = np.array(s, dtype=np.float64, order='C', copy=True)
    w = _initialize_w(s, rank, "random_sqrt", seed)
    w = np.array(w, dtype=np.float64, order='C', copy=True)

    n = s.shape[0]
    s_norm = np.linalg.norm(s, "fro")

    for it in range(max_outer):
        w = bsum_update(s, w, max_iter=max_inner, tol=tol)
        w = np.array(w, dtype=np.float64, order='C', copy=True)

        if target_sp > 0:
            w = project_w(w, target_sp)
            w = np.array(w, dtype=np.float64, order='C', copy=True)

        res_norm, xhat_norm = _frobenius_residual(s, w)
        eps_pri = n * tol + 1e-4 * max(s_norm, xhat_norm)
        if res_norm <= eps_pri:
            break

    return w


def _fit_eval(s, rank, seed, val, target_sp, name):
    w = fit_hoyer_srf(s, rank, seed, target_sp)
    sp = np.mean([hoyer_sparseness(w[i]) for i in range(w.shape[0])])
    frac_zero = (w < 1e-4).mean()
    return name, seed, acc(w, val), sp, frac_zero


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
    vice_accs, vice_sp = [], []
    for seed in range(10):
        d = np.load(
            vice_dir / f"vice_100pct_part0_seed{seed}" / "variational" / "4.12mio" / "sslab" / "90" / "256" / "1.0" / str(seed) / "params" / "parameters.npz",
            allow_pickle=True,
        )
        w = np.maximum(d["pruned_q_mu"], 0)
        vice_accs.append(acc(w, val))
        vice_sp.append(np.mean([hoyer_sparseness(w[i]) for i in range(w.shape[0])]))
    log.info(f"VICE: {np.mean(vice_accs):.4f}, hoyer_sp={np.mean(vice_sp):.3f}, zeros={(w<1e-4).mean():.3f}")

    # Sweep target sparseness
    targets = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    conditions = [(s, rank, seed, val, sp, f"sp={sp}")
                  for sp in targets for seed in range(3)]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_eval)(*c) for c in conditions
    )

    log.info("\nResults:")
    for sp in targets:
        sub = [(a, h, z) for n, sd, a, h, z in results if n == f"sp={sp}"]
        accs_list = [a for a, h, z in sub]
        hoyer_list = [h for a, h, z in sub]
        zero_list = [z for a, h, z in sub]
        log.info(f"  sp={sp:.1f}: acc={np.mean(accs_list):.4f}, "
                 f"hoyer={np.mean(hoyer_list):.3f}, zeros={np.mean(zero_list):.3f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
