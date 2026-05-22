"""EM-like iterative blending: continuous reweighting without changing pysrf.

Alternate between:
  M-step: W = SRF(S_blended)
  E-step: S_blended = confidence * S_data + (1 - confidence) * WW^T
  where confidence = shown / (shown + lambda)

Lambda controls per-entry trust. High-confidence entries stay close to data.
Low-confidence entries lean on the model's low-rank reconstruction.
"""
import json
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
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


def main():
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)

    with open(KAPPA_RANKS) as f:
        rank = json.load(f)["100"]
    log.info(f"Kappa rank: {rank}")

    # Build count RSM (alpha=0, matches existing experiments)
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1); np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1); np.add.at(shown, (b, a), 1)

    s_data = np.divide(counts, shown, out=np.full((N, N), np.nan), where=shown > 0)
    np.fill_diagonal(s_data, 1.0)

    # Fill NaN with 0.5 for initial fit
    s_data_filled = s_data.copy()
    s_data_filled[np.isnan(s_data_filled)] = 0.5

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
    log.info(f"VICE: {np.mean(vice_accs):.4f} +/- {np.std(vice_accs):.4f}")

    # SRF baseline (no blending)
    w_base = SRF(rank=rank, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0).fit_transform(s_data_filled)
    log.info(f"SRF baseline: {acc(w_base, val):.4f}")

    # EM blending
    observed = shown > 0

    for lam in [1.0, 3.0, 5.0, 10.0, 20.0]:
        log.info(f"\nlambda={lam}:")
        confidence = shown / (shown + lam)
        confidence[~observed] = 0.0

        s_blend = s_data_filled.copy()

        for it in range(10):
            w = SRF(rank=rank, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0).fit_transform(s_blend)
            s_model = w @ w.T

            # Normalize s_model diagonal to 1
            d = np.sqrt(np.diag(s_model))
            d[d == 0] = 1
            s_model = s_model / np.outer(d, d)

            # Blend
            s_blend = confidence * s_data_filled + (1 - confidence) * s_model
            np.fill_diagonal(s_blend, 1.0)

            a = acc(w, val)
            log.info(f"  iter {it}: {a:.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
