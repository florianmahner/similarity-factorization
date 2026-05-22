"""Test PPMI-based RSM from triplets: does adjusting for item popularity help?

count RSM: S[i,j] = count(i,j) / shown(i,j)
PPMI RSM: PPMI(i,j) = max(0, log(P(i,j) / (P(i)*P(j))))
  where P(i,j) = count(i,j) / N, P(i) = sum_j count(i,j) / (2*N)
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


def build_count_rsm(train, n):
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    np.add.at(counts, (ii, jj), 1); np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1); np.add.at(shown, (b, a), 1)
    return counts, shown


def build_ppmi(counts, alpha_smooth=0.0):
    """PPMI from the chosen-pair count matrix."""
    c = counts + alpha_smooth
    total = c.sum() / 2  # each pair counted twice (symmetric)
    p_ij = c / (2 * total)  # joint probability
    p_i = c.sum(axis=1) / (2 * total)  # marginal
    p_ip = np.outer(p_i, p_i)
    pmi = np.log(p_ij / (p_ip + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi, 0)
    np.fill_diagonal(ppmi, 0)
    # Normalize to [0, 1] with diagonal = 1
    if ppmi.max() > 0:
        ppmi = ppmi / ppmi.max()
    np.fill_diagonal(ppmi, 1.0)
    return ppmi


def build_shifted_ppmi(counts, k_shift=1.0, alpha_smooth=0.0):
    """Shifted PPMI: PPMI - log(k), following Levy & Goldberg (2014)."""
    c = counts + alpha_smooth
    total = c.sum() / 2
    p_ij = c / (2 * total)
    p_i = c.sum(axis=1) / (2 * total)
    p_ip = np.outer(p_i, p_i)
    pmi = np.log(p_ij / (p_ip + 1e-12) + 1e-12)
    sppmi = np.maximum(pmi - np.log(k_shift), 0)
    np.fill_diagonal(sppmi, 0)
    if sppmi.max() > 0:
        sppmi = sppmi / sppmi.max()
    np.fill_diagonal(sppmi, 1.0)
    return sppmi


def _fit_eval(rsm, rank, seed, val, name):
    model = SRF(rank=rank, random_state=seed, max_outer=500, max_inner=30, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm)
    return name, seed, acc(w, val)


def main():
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)
    rank = json.load(open(KAPPA_RANKS))["100"]
    log.info(f"Rank: {rank}")

    counts, shown = build_count_rsm(train, N)

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

    # Build RSM variants
    s_count = np.divide(counts, shown, out=np.full((N, N), 0.5), where=shown > 0)
    np.fill_diagonal(s_count, 1.0)

    s_ppmi = build_ppmi(counts)
    s_ppmi_smooth = build_ppmi(counts, alpha_smooth=1.0)
    s_sppmi_2 = build_shifted_ppmi(counts, k_shift=2.0)
    s_sppmi_5 = build_shifted_ppmi(counts, k_shift=5.0)

    log.info(f"\nRSM stats:")
    for name, rsm in [("count", s_count), ("ppmi", s_ppmi), ("ppmi_s1", s_ppmi_smooth),
                       ("sppmi_k2", s_sppmi_2), ("sppmi_k5", s_sppmi_5)]:
        off = rsm[~np.eye(N, dtype=bool)]
        log.info(f"  {name:10s}: mean={off.mean():.4f}, std={off.std():.4f}, "
                 f"zeros={(off==0).mean():.4f}, NaN={np.isnan(rsm).sum()}")

    # Fit all in parallel (5 RSMs x 3 seeds = 15 fits)
    conditions = []
    for name, rsm in [("count", s_count), ("ppmi", s_ppmi), ("ppmi_s1", s_ppmi_smooth),
                       ("sppmi_k2", s_sppmi_2), ("sppmi_k5", s_sppmi_5)]:
        for seed in range(3):
            conditions.append((rsm, rank, seed, val, name))

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_eval)(*c) for c in conditions
    )

    for name in ["count", "ppmi", "ppmi_s1", "sppmi_k2", "sppmi_k5"]:
        accs_list = [a for n, s, a in results if n == name]
        log.info(f"  {name:10s}: {np.mean(accs_list):.4f} +/- {np.std(accs_list):.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
