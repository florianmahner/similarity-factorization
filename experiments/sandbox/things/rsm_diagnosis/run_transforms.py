"""Test monotone RSM transforms that expand the high-similarity range."""
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


def normalize_rsm(s):
    """Normalize off-diagonal to [0, 1], diagonal = 1."""
    s = s.copy()
    np.fill_diagonal(s, np.nan)
    lo, hi = np.nanmin(s), np.nanmax(s)
    s = (s - lo) / (hi - lo)
    np.fill_diagonal(s, 1.0)
    return s


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
    np.fill_diagonal(s_raw, 1.0)

    # Build transform variants
    transforms = {}

    # Baseline
    transforms["count"] = s_raw

    # -log(1 - S): expands high similarity range
    s_nlog = -np.log(1 - np.clip(s_raw, 0, 0.999))
    transforms["neglog"] = normalize_rsm(s_nlog)

    # S^2: mild expansion of high end
    transforms["sq"] = normalize_rsm(s_raw ** 2)

    # S^0.5: expansion of low end (opposite direction, as control)
    transforms["sqrt"] = normalize_rsm(np.sqrt(s_raw))

    # logit: symmetric expansion of both tails
    s_logit = np.log(np.clip(s_raw, 0.01, 0.99) / (1 - np.clip(s_raw, 0.01, 0.99)))
    transforms["logit"] = normalize_rsm(s_logit)

    # exp(S) - 1: another monotone expansion of high end
    transforms["exp"] = normalize_rsm(np.exp(s_raw) - 1)

    # S / (1-S): odds ratio, strong expansion of high end
    s_odds = s_raw / (1 - np.clip(s_raw, 0, 0.999))
    transforms["odds"] = normalize_rsm(s_odds)

    # -log(1-S) without normalization (keep expanded scale)
    s_nlog_raw = -np.log(1 - np.clip(s_raw, 0, 0.999))
    np.fill_diagonal(s_nlog_raw, s_nlog_raw[~np.eye(N, dtype=bool)].max() * 1.1)
    transforms["neglog_raw"] = s_nlog_raw

    for name, s in transforms.items():
        off = s[~np.eye(N, dtype=bool)]
        log.info(f"  {name:12s}: range=[{off.min():.3f}, {off.max():.3f}], mean={off.mean():.3f}")

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
    log.info(f"\n  VICE: {np.mean(vice_accs):.4f} +/- {np.std(vice_accs):.4f}")

    # Fit all in parallel (8 transforms x 3 seeds = 24 fits)
    conditions = [(s, rank, seed, val, name)
                  for name, s in transforms.items()
                  for seed in range(3)]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit)(*c) for c in conditions
    )

    log.info("\nResults:")
    for name in transforms:
        accs_list = [a for n, s, a in results if n == name]
        log.info(f"  {name:12s}: {np.mean(accs_list):.4f} +/- {np.std(accs_list):.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
