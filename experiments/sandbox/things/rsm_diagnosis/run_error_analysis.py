"""Error analysis: which triplets does SRF get wrong that VICE gets right, and why?"""
import json
import logging
from pathlib import Path

import numpy as np

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
KAPPA_RANKS = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "srf" / "outputs" / "kappa_alpha0" / "ranks.json"
N = 1854


def predict(w, trips):
    """Return per-triplet correctness and margins."""
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    correct = (sij > sik) & (sij > sjk)
    margin = sij - np.maximum(sik, sjk)
    return correct, margin, sij, sik, sjk


def main():
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)
    rank = json.load(open(KAPPA_RANKS))["100"]

    # Build count RSM for context
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1); np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1); np.add.at(shown, (b, a), 1)
    s_count = np.divide(counts, shown, out=np.full((N, N), 0.5), where=shown > 0)

    # Load SRF (kappa, seed 0)
    srf_dir = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "srf" / "outputs" / "kappa_alpha0" / "embeddings"
    w_srf = np.load(srf_dir / "srf_100pct_part0_seed0.npz")["embedding"]

    # Load VICE (seed 0)
    vice_dir = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "vice" / "outputs" / "models"
    d = np.load(
        vice_dir / "vice_100pct_part0_seed0" / "variational" / "4.12mio" / "sslab" / "90" / "256" / "1.0" / "0" / "params" / "parameters.npz",
        allow_pickle=True,
    )
    w_vice = np.maximum(d["pruned_q_mu"], 0)

    # Predictions
    srf_ok, srf_margin, srf_sij, srf_sik, srf_sjk = predict(w_srf, val)
    vice_ok, vice_margin, vice_sij, vice_sik, vice_sjk = predict(w_vice, val)

    log.info(f"SRF:  {srf_ok.mean():.4f} ({srf_ok.sum():,}/{len(val):,})")
    log.info(f"VICE: {vice_ok.mean():.4f} ({vice_ok.sum():,}/{len(val):,})")

    # Contingency
    both_right = srf_ok & vice_ok
    srf_only = srf_ok & ~vice_ok
    vice_only = ~srf_ok & vice_ok
    both_wrong = ~srf_ok & ~vice_ok

    log.info(f"\nContingency:")
    log.info(f"  Both right:  {both_right.sum():>7,} ({both_right.mean()*100:.1f}%)")
    log.info(f"  SRF only:    {srf_only.sum():>7,} ({srf_only.mean()*100:.1f}%)")
    log.info(f"  VICE only:   {vice_only.sum():>7,} ({vice_only.mean()*100:.1f}%)")
    log.info(f"  Both wrong:  {both_wrong.sum():>7,} ({both_wrong.mean()*100:.1f}%)")

    # Analyze VICE-only triplets (what SRF gets wrong that VICE gets right)
    log.info(f"\n{'='*60}")
    log.info("VICE-only triplets (SRF wrong, VICE right)")
    log.info(f"{'='*60}")

    vo_idx = np.where(vice_only)[0]
    vo_trips = val[vo_idx]

    # RSM properties for these triplets
    vo_s_ij = s_count[vo_trips[:, 0], vo_trips[:, 1]]
    vo_s_ik = s_count[vo_trips[:, 0], vo_trips[:, 2]]
    vo_s_jk = s_count[vo_trips[:, 1], vo_trips[:, 2]]
    vo_shown_ij = shown[vo_trips[:, 0], vo_trips[:, 1]]
    vo_shown_ik = shown[vo_trips[:, 0], vo_trips[:, 2]]
    vo_shown_jk = shown[vo_trips[:, 1], vo_trips[:, 2]]

    # Same for all triplets
    all_s_ij = s_count[val[:, 0], val[:, 1]]
    all_s_ik = s_count[val[:, 0], val[:, 2]]
    all_s_jk = s_count[val[:, 1], val[:, 2]]
    all_shown_ij = shown[val[:, 0], val[:, 1]]

    rsm_margin_all = all_s_ij - np.maximum(all_s_ik, all_s_jk)
    rsm_margin_vo = vo_s_ij - np.maximum(vo_s_ik, vo_s_jk)

    log.info(f"\n  RSM margin (s_ij - max(s_ik, s_jk)):")
    log.info(f"    All triplets:  mean={rsm_margin_all.mean():.4f}, std={rsm_margin_all.std():.4f}")
    log.info(f"    VICE-only:     mean={rsm_margin_vo.mean():.4f}, std={rsm_margin_vo.std():.4f}")

    log.info(f"\n  RSM values (correct pair s_ij):")
    log.info(f"    All triplets:  mean={all_s_ij.mean():.4f}")
    log.info(f"    VICE-only:     mean={vo_s_ij.mean():.4f}")

    log.info(f"\n  Observation counts (shown for correct pair):")
    log.info(f"    All triplets:  mean={all_shown_ij.mean():.1f}")
    log.info(f"    VICE-only:     mean={vo_shown_ij.mean():.1f}")

    log.info(f"\n  SRF margins on VICE-only triplets:")
    log.info(f"    mean={srf_margin[vo_idx].mean():.4f}, std={srf_margin[vo_idx].std():.4f}")
    log.info(f"    fraction with SRF margin in [-0.01, 0]: {((srf_margin[vo_idx] > -0.01) & (srf_margin[vo_idx] < 0)).mean():.3f}")
    log.info(f"    fraction with SRF margin < -0.1: {(srf_margin[vo_idx] < -0.1).mean():.3f}")

    log.info(f"\n  VICE margins on VICE-only triplets:")
    log.info(f"    mean={vice_margin[vo_idx].mean():.4f}, std={vice_margin[vo_idx].std():.4f}")

    # What does the RSM say for these triplets?
    rsm_correct_vo = (vo_s_ij > vo_s_ik) & (vo_s_ij > vo_s_jk)
    log.info(f"\n  RSM predicts correctly on VICE-only: {rsm_correct_vo.mean():.3f}")

    # Breakdown by RSM margin buckets
    log.info(f"\n  VICE-only by RSM margin bucket:")
    for lo, hi in [(-1, -0.1), (-0.1, -0.01), (-0.01, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 1)]:
        mask = (rsm_margin_vo >= lo) & (rsm_margin_vo < hi)
        if mask.sum() > 10:
            log.info(f"    [{lo:+.2f}, {hi:+.2f}): n={mask.sum():>5}, "
                     f"SRF margin={srf_margin[vo_idx][mask].mean():.4f}, "
                     f"VICE margin={vice_margin[vo_idx][mask].mean():.4f}")

    # Analyze by object frequency
    log.info(f"\n  Object properties:")
    obj_shown = shown.sum(axis=1)
    obj_count = counts.sum(axis=1)
    obj_popularity = obj_count / obj_shown  # how often this object is in the chosen pair

    i_pop = obj_popularity[vo_trips[:, 0]]
    j_pop = obj_popularity[vo_trips[:, 1]]
    k_pop = obj_popularity[vo_trips[:, 2]]
    all_i_pop = obj_popularity[val[:, 0]]

    log.info(f"    Object popularity (fraction of times in chosen pair):")
    log.info(f"      All anchor i: mean={all_i_pop.mean():.4f}")
    log.info(f"      VICE-only anchor i: mean={i_pop.mean():.4f}")
    log.info(f"      VICE-only j (chosen): mean={j_pop.mean():.4f}")
    log.info(f"      VICE-only k (odd-out): mean={k_pop.mean():.4f}")

    # Embedding norms
    srf_norms = np.linalg.norm(w_srf, axis=1)
    vice_norms = np.linalg.norm(w_vice, axis=1)

    log.info(f"\n  Embedding norms for objects in VICE-only triplets:")
    log.info(f"    SRF  i: {srf_norms[vo_trips[:, 0]].mean():.3f}, j: {srf_norms[vo_trips[:, 1]].mean():.3f}, k: {srf_norms[vo_trips[:, 2]].mean():.3f}")
    log.info(f"    VICE i: {vice_norms[vo_trips[:, 0]].mean():.3f}, j: {vice_norms[vo_trips[:, 1]].mean():.3f}, k: {vice_norms[vo_trips[:, 2]].mean():.3f}")
    log.info(f"    SRF  all: {srf_norms.mean():.3f} +/- {srf_norms.std():.3f}")
    log.info(f"    VICE all: {vice_norms.mean():.3f} +/- {vice_norms.std():.3f}")

    # Sparsity comparison
    srf_sparsity = (w_srf < 1e-4).mean(axis=1)
    vice_sparsity = (w_vice < 1e-4).mean(axis=1)
    log.info(f"\n  Sparsity (fraction of near-zero dims):")
    log.info(f"    SRF:  {srf_sparsity.mean():.3f}")
    log.info(f"    VICE: {vice_sparsity.mean():.3f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
