"""Test diagonal treatment and robust RSM construction."""
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


def _fit(rsm, rank, seed, val, name):
    model = SRF(rank=rank, random_state=seed, max_outer=500, max_inner=30, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm)
    return name, seed, acc(w, val)


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

    s_raw = np.divide(counts, shown, out=np.full((N, N), 0.5), where=shown > 0)

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

    # Check VICE diagonal variance
    d = np.load(
        vice_dir / "vice_100pct_part0_seed0" / "variational" / "4.12mio" / "sslab" / "90" / "256" / "1.0" / "0" / "params" / "parameters.npz",
        allow_pickle=True,
    )
    w_vice = np.maximum(d["pruned_q_mu"], 0)
    vice_norms = np.sum(w_vice ** 2, axis=1)
    log.info(f"VICE ||w_i||^2: mean={vice_norms.mean():.3f}, std={vice_norms.std():.3f}, "
             f"min={vice_norms.min():.3f}, max={vice_norms.max():.3f}")

    variants = {}

    # 1. Baseline: diagonal = 1
    s1 = s_raw.copy()
    np.fill_diagonal(s1, 1.0)
    variants["diag=1.0"] = s1

    # 2. Diagonal = NaN (SRF imputes from model)
    s2 = s_raw.copy()
    np.fill_diagonal(s2, np.nan)
    variants["diag=NaN"] = s2

    # 3. Diagonal = mean of row (data-driven per object)
    s3 = s_raw.copy()
    row_means = np.nanmean(s_raw, axis=1)
    np.fill_diagonal(s3, row_means * 2)  # self-sim > mean cross-sim
    variants["diag=2*rowmean"] = s3

    # 4. Diagonal = max of row
    s4 = s_raw.copy()
    row_max = np.nanmax(s_raw, axis=1)
    np.fill_diagonal(s4, row_max * 1.1)
    variants["diag=1.1*rowmax"] = s4

    # 5. No diagonal constraint: set to large value so SRF ignores relative to off-diag
    s5 = s_raw.copy()
    np.fill_diagonal(s5, 10.0)
    variants["diag=10"] = s5

    # 6. Robust: fit SRF, identify outliers, suppress, refit
    s6 = s_raw.copy()
    np.fill_diagonal(s6, 1.0)
    w_init = SRF(rank=rank, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0).fit_transform(s6)
    s_model = w_init @ w_init.T
    d_model = np.sqrt(np.diag(s_model)); d_model[d_model==0]=1
    s_model = s_model / np.outer(d_model, d_model)
    residual = np.abs(s6 - s_model)
    threshold = np.percentile(residual[~np.eye(N, dtype=bool)], 95)
    outlier = residual > threshold
    s6_robust = s6.copy()
    s6_robust[outlier] = s_model[outlier]  # replace outliers with model prediction
    variants["robust_95pct"] = s6_robust

    # 7. Even more aggressive: 90th percentile
    threshold_90 = np.percentile(residual[~np.eye(N, dtype=bool)], 90)
    outlier_90 = residual > threshold_90
    s7 = s6.copy()
    s7[outlier_90] = s_model[outlier_90]
    variants["robust_90pct"] = s7

    # Fit all in parallel
    conditions = [(s, rank, seed, val, name)
                  for name, s in variants.items()
                  for seed in range(3)]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit)(*c) for c in conditions
    )

    log.info("\nResults:")
    for name in variants:
        accs_list = [a for n, s, a in results if n == name]
        log.info(f"  {name:18s}: {np.mean(accs_list):.4f} +/- {np.std(accs_list):.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
